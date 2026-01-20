"""
Reflexion Strategy - Learning from mistakes through self-reflection.

Reflexion follows a "try, reflect on failure, retry with insight" pattern:
1. Attempt task execution
2. On failure, reflect on what went wrong
3. Store insight in memory
4. Retry with accumulated insights as context

Key insight: Learn from mistakes rather than just decomposing problems.
The agent builds a memory of failure patterns and corrections.

Reference: "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
"""

import typing as t
from dataclasses import dataclass, field

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agents.agent_tool.planning_strategies import PlanningStrategy, StrategyOutput
from agents.configs import get_agent_tool_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool

# Load templates
_templates = get_agent_tool_template_module("reflexion_strategy.jinja")


class ReflectionInsight(BaseModel):
    """A single reflection insight from a failure."""

    iteration: int = Field(description="Which iteration this insight came from")
    failure_description: str = Field(description="What went wrong")
    insight: str = Field(description="What was learned / how to avoid this")
    tool_involved: str | None = Field(
        default=None, description="Tool that failed, if any"
    )


@dataclass
class ReflexionMemory:
    """
    Memory store for Reflexion strategy.

    Maintains insights across iterations within a task,
    and optionally across tasks (cross-task learning).
    """

    # Insights from current task
    task_insights: list[ReflectionInsight] = field(default_factory=list)

    # Persistent insights across tasks (optional)
    persistent_insights: list[ReflectionInsight] = field(default_factory=list)

    # Track consecutive failures for circuit breaker
    consecutive_failures: int = 0

    def add_insight(self, insight: ReflectionInsight, persist: bool = False) -> None:
        """Add a new insight to memory."""
        self.task_insights.append(insight)
        if persist:
            self.persistent_insights.append(insight)

    def get_context_prompt(self, max_insights: int = 5) -> str:
        """Generate context prompt from accumulated insights."""
        if not self.task_insights and not self.persistent_insights:
            return ""

        lines = ["## Lessons from Previous Attempts\n"]

        # Include persistent insights first (cross-task learning)
        if self.persistent_insights:
            lines.append("### General Patterns to Avoid:")
            for insight in self.persistent_insights[-max_insights:]:
                lines.append(f"- {insight.insight}")
            lines.append("")

        # Include task-specific insights
        if self.task_insights:
            lines.append("### This Task's Learnings:")
            for insight in self.task_insights[-max_insights:]:
                lines.append(f"- Attempt {insight.iteration}: {insight.insight}")

        return "\n".join(lines)

    def reset_task(self) -> None:
        """Reset task-specific memory (keep persistent)."""
        self.task_insights = []
        self.consecutive_failures = 0

    def reset_all(self) -> None:
        """Full reset including persistent memory."""
        self.task_insights = []
        self.persistent_insights = []
        self.consecutive_failures = 0


class ReflexionStrategy(PlanningStrategy):
    """
    Reflexion Strategy: Learn from failures through self-reflection.

    Flow per iteration:
        1. Inject accumulated insights as context
        2. Execute action (tool selection)
        3. Detect failure from results
        4. If failed: Generate reflection → Store insight → Retry
        5. If succeeded: Continue or finish

    Configuration:
        - max_reflections: Maximum reflection cycles before giving up
        - reflection_threshold: Error severity triggering reflection
        - persist_insights: Whether to keep insights across tasks

    Memory:
        - task_insights: Learnings specific to current task
        - persistent_insights: Cross-task patterns (optional)

    Return behavior matches base strategies:
        - No tool_calls → finished=True, success=False
        - FINISH tool called → finished=True, success=<from args>
        - Other tools called → finished=False, tool_calls=[...]
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str | None = None,
        max_reflections: int = 3,
        persist_insights: bool = False,
        finish_tool_name: str = "finish",
        action_prompt: str | None = None,
        reflection_prompt: str | None = None,
    ):
        """
        Initialize Reflexion Strategy.

        Args:
            llm_client: LLM client for action and reflection phases
            model: Optional model override
            max_reflections: Maximum reflection cycles before giving up
            persist_insights: Keep insights across tasks (cross-task learning)
            finish_tool_name: Name of the finish tool
            action_prompt: Custom prompt for action phase
            reflection_prompt: Custom prompt for reflection phase
        """
        self.llm_client = llm_client
        self.model = model
        self.max_reflections = max_reflections
        self.persist_insights = persist_insights
        self.finish_tool_name = finish_tool_name
        self.action_prompt = action_prompt
        self.reflection_prompt = reflection_prompt

        # State
        self.memory = ReflexionMemory()
        self._iteration = 0
        self._pending_reflection = False
        self._last_failure_context: str | None = None

    def _get_action_prompt(self, tool_names: list[str]) -> str:
        """Get prompt for action phase."""
        if self.action_prompt:
            return self.action_prompt
        return _templates.action_prompt(tool_names=tool_names)

    def _get_reflection_prompt(self, failure_context: str) -> str:
        """Get prompt for reflection phase."""
        if self.reflection_prompt:
            return self.reflection_prompt
        return _templates.reflection_prompt(failure_context=failure_context)

    def _detect_failure_in_messages(
        self, messages: list[ChatCompletionMessageParam]
    ) -> tuple[bool, str]:
        """
        Detect failure signals from recent messages.

        Returns:
            (failure_detected, failure_description)
        """
        failure_signals: list[str] = []

        # Look at recent tool results (last few messages)
        recent_tool_results = [
            msg for msg in messages[-6:] if msg.get("role") == "tool"
        ]

        for msg in recent_tool_results:
            content = str(msg.get("content", ""))

            # Check for error patterns
            if '"error"' in content.lower():
                # Extract error message if possible
                try:
                    import json

                    data = json.loads(content)
                    if "error" in data:
                        failure_signals.append(f"Tool error: {data['error']}")
                except (json.JSONDecodeError, KeyError):
                    failure_signals.append("Tool returned an error")

            elif "error:" in content.lower():
                failure_signals.append(f"Error in result: {content[:100]}")

            elif "failed" in content.lower() or "failure" in content.lower():
                failure_signals.append(f"Failure indication: {content[:100]}")

        if failure_signals:
            return True, "; ".join(failure_signals)

        return False, ""

    async def _generate_reflection(
        self,
        messages: list[ChatCompletionMessageParam],
        failure_context: str,
    ) -> ReflectionInsight:
        """
        Generate a reflection insight from failure.

        Uses LLM to analyze what went wrong and extract a lesson.
        """
        logger.info(
            "Reflexion: Generating reflection | iteration={} | failure={}",
            self._iteration,
            failure_context[:50],
        )

        reflection_messages = list(messages) + [
            {"role": "user", "content": self._get_reflection_prompt(failure_context)}
        ]

        response = await self.llm_client.agenerate(
            messages=reflection_messages,
            model=self.model,
            mode="text",
        )

        insight_text = response.content or "Unable to generate insight"

        # Extract tool name if mentioned in failure
        tool_involved = None
        for word in failure_context.split():
            if word.isupper() and len(word) > 2:
                tool_involved = word
                break

        return ReflectionInsight(
            iteration=self._iteration,
            failure_description=failure_context,
            insight=insight_text,
            tool_involved=tool_involved,
        )

    async def plan(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool = True,
    ) -> StrategyOutput:
        """
        Generate next actions using Reflexion pattern.

        Flow:
            1. Check for pending reflection from previous failure
            2. If pending: generate insight, store in memory
            3. Inject accumulated insights into context
            4. Execute action phase with enhanced context
            5. Detect failure for next iteration
        """
        self._iteration += 1
        tool_names = [t.name for t in tools]

        # Step 1: Handle pending reflection from previous iteration
        if self._pending_reflection and self._last_failure_context:
            if self.memory.consecutive_failures >= self.max_reflections:
                logger.warning(
                    "Reflexion: Max reflections reached | count={}",
                    self.max_reflections,
                )
                return StrategyOutput(
                    finished=True,
                    success=False,
                    result=f"Failed after {self.max_reflections} reflection attempts. "
                    f"Last failure: {self._last_failure_context}",
                )

            # Generate and store reflection
            insight = await self._generate_reflection(
                messages, self._last_failure_context
            )
            self.memory.add_insight(insight, persist=self.persist_insights)
            self.memory.consecutive_failures += 1

            logger.debug(
                "Reflexion: Insight stored | insight={}",
                insight.insight[:100],
            )

            self._pending_reflection = False
            self._last_failure_context = None

        # Step 2: Build context with accumulated insights
        insight_context = self.memory.get_context_prompt()

        # Step 3: Execute action phase
        action_messages = list(messages)

        # Inject insights as system context if we have any
        if insight_context:
            action_messages.append({"role": "system", "content": insight_context})

        action_messages.append(
            {"role": "user", "content": self._get_action_prompt(tool_names)}
        )

        logger.debug(
            "Reflexion: Action phase | iteration={} | insights={}",
            self._iteration,
            len(self.memory.task_insights),
        )

        response = await self.llm_client.agenerate(
            messages=action_messages,
            model=self.model,
            mode="tool_calling",
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Step 4: Process response
        result = self._process_response(response)

        # Step 5: Check for failure to trigger reflection next iteration
        if not result.finished:
            # We'll check for failure after tools execute (next plan() call)
            failure_detected, failure_desc = self._detect_failure_in_messages(messages)
            if failure_detected:
                self._pending_reflection = True
                self._last_failure_context = failure_desc
        else:
            # Task finished - reset consecutive failures on success
            if result.success:
                self.memory.consecutive_failures = 0

        return result

    def _process_response(self, response: t.Any) -> StrategyOutput:
        """Process LLM response into StrategyOutput."""
        # No tool calls → unsuccessful finish
        if not response.tool_calls:
            return StrategyOutput(
                finished=True,
                success=False,
                result=response.finish_reason or "No tool calls returned by LLM",
            )

        # Check for finish tool
        for tc in response.tool_calls:
            if tc.tool_name.upper() == self.finish_tool_name.upper():
                return StrategyOutput(
                    finished=True,
                    success=tc.arguments.get("success", True),
                    result=tc.arguments.get("result", "Task completed"),
                )

        # Return tool calls for execution
        return StrategyOutput(tool_calls=response.tool_calls)

    def reset(self, full: bool = False) -> None:
        """
        Reset strategy state.

        Args:
            full: If True, also clear persistent insights
        """
        self._iteration = 0
        self._pending_reflection = False
        self._last_failure_context = None

        if full:
            self.memory.reset_all()
        else:
            self.memory.reset_task()

    def get_insights(self) -> list[ReflectionInsight]:
        """Get all accumulated insights (for inspection/debugging)."""
        return self.memory.task_insights + self.memory.persistent_insights

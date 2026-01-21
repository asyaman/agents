"""
Adaptive Reflexion Strategy - Combining ADaPT and Reflexion patterns.

This strategy combines two complementary failure recovery mechanisms:
1. Reflexion: Learn from mistakes through self-reflection (behavioral adaptation)
2. ADaPT: Decompose complex tasks when simple approaches fail (structural adaptation)

Flow:
    DIRECT → (failure) → REFLECT → RETRY → (still failing) → DECOMPOSE → EXECUTE_SUBTASKS
                ↑                    ↓
                └────────────────────┘ (reflection loop)

Key insight: First try to learn and correct behavior. If behavioral changes
don't work, the problem may be structural and needs decomposition.

This provides a more robust failure recovery than either approach alone.
"""

import typing as t
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam

from agents.agent_tool.base_strategy import PlanningStrategy, StrategyOutput
from agents.agent_tool.reflexion_strategy import ReflectionInsight, ReflexionMemory
from agents.configs import get_agent_tool_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool

# Load templates
_templates = get_agent_tool_template_module("adaptive_reflexion_strategy.jinja")


class AdaptivePhase(str, Enum):
    """Current phase of the adaptive reflexion strategy."""

    DIRECT = "direct"  # Initial direct execution attempt
    REFLECTING = "reflecting"  # Analyzing failure and generating insight
    RETRYING = "retrying"  # Retrying with accumulated insights
    DECOMPOSING = "decomposing"  # Breaking down into subtasks
    EXECUTING_SUBTASKS = "executing_subtasks"  # Running decomposed subtasks


@dataclass
class AdaptiveState:
    """
    State tracking for the adaptive reflexion strategy.

    Tracks both reflection-based learning and decomposition triggers.
    """

    phase: AdaptivePhase = AdaptivePhase.DIRECT
    iteration: int = 0
    direct_attempts: int = 0
    reflection_cycles: int = 0
    decomposition_triggered: bool = False

    # Failure tracking
    pending_reflection: bool = False
    last_failure_context: str = ""
    consecutive_failures: int = 0

    # Reflection memory (inherited from Reflexion)
    memory: ReflexionMemory = field(default_factory=ReflexionMemory)

    def reset(self) -> None:
        """Reset state for new task."""
        self.phase = AdaptivePhase.DIRECT
        self.iteration = 0
        self.direct_attempts = 0
        self.reflection_cycles = 0
        self.decomposition_triggered = False
        self.pending_reflection = False
        self.last_failure_context = ""
        self.consecutive_failures = 0
        self.memory.reset_task()


class AdaptiveReflexionStrategy(PlanningStrategy):
    """
    Combined ADaPT + Reflexion Strategy.

    This strategy provides multi-layered failure recovery:

    Layer 1 - Direct Execution:
        Try to complete the task directly with available tools.

    Layer 2 - Reflexion Loop:
        On failure, reflect on what went wrong, store insight, and retry.
        This handles behavioral issues (wrong approach, missing steps, etc.)

    Layer 3 - ADaPT Decomposition:
        If reflection doesn't resolve the issue after max_reflections,
        decompose the task into subtasks. This handles structural issues
        (task too complex, needs parallel workstreams, etc.)

    Configuration:
        - max_direct_attempts: Attempts before first reflection
        - max_reflections: Reflection cycles before decomposition
        - max_subtask_depth: Nested decomposition limit

    The strategy transitions through phases:
        DIRECT → REFLECTING → RETRYING → DECOMPOSING → EXECUTING_SUBTASKS
                     ↑           ↓
                     └───────────┘ (loop until max_reflections)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str | None = None,
        max_direct_attempts: int = 2,
        max_reflections: int = 2,
        persist_insights: bool = False,
        finish_tool_name: str = "finish",
        sub_agent_tool_name: str = "delegate_subtask",
    ):
        """
        Initialize Adaptive Reflexion Strategy.

        Args:
            llm_client: LLM client for all phases
            model: Optional model override
            max_direct_attempts: Attempts before first reflection
            max_reflections: Max reflection cycles before decomposition
            persist_insights: Keep insights across tasks
            finish_tool_name: Name of the finish tool
            sub_agent_tool_name: Name of the SubAgentTool
        """
        self.llm_client = llm_client
        self.model = model
        self.max_direct_attempts = max_direct_attempts
        self.max_reflections = max_reflections
        self.persist_insights = persist_insights
        self.finish_tool_name = finish_tool_name
        self.sub_agent_tool_name = sub_agent_tool_name

        # State
        self._state = AdaptiveState()

    def _has_sub_agent_tool(self, tools: list[BaseTool[t.Any, t.Any]]) -> bool:
        """Check if SubAgentTool is available."""
        return any(t.name.lower() == self.sub_agent_tool_name.lower() for t in tools)

    def _filter_tools(
        self, tools: list[BaseTool[t.Any, t.Any]], include_sub_agent: bool
    ) -> list[BaseTool[t.Any, t.Any]]:
        """Filter tools based on whether to include SubAgentTool."""
        if include_sub_agent:
            return list(tools)
        return [t for t in tools if t.name.lower() != self.sub_agent_tool_name.lower()]

    def _detect_failure(
        self, messages: list[ChatCompletionMessageParam]
    ) -> tuple[bool, str]:
        """Detect failure signals from recent messages."""
        failure_signals: list[str] = []

        recent_tool_results = [
            msg for msg in messages[-6:] if msg.get("role") == "tool"
        ]

        for msg in recent_tool_results:
            content = str(msg.get("content", ""))

            if '"error"' in content.lower():
                try:
                    import json

                    data = json.loads(content)
                    if "error" in data:
                        failure_signals.append(f"Tool error: {data['error']}")
                except (json.JSONDecodeError, KeyError):
                    failure_signals.append("Tool returned an error")
            elif "error:" in content.lower():
                failure_signals.append(f"Error: {content[:100]}")
            elif "failed" in content.lower():
                failure_signals.append(f"Failure: {content[:100]}")

        if failure_signals:
            return True, "; ".join(failure_signals)

        return False, ""

    async def _generate_reflection(
        self,
        messages: list[ChatCompletionMessageParam],
        failure_context: str,
    ) -> ReflectionInsight:
        """Generate a reflection insight from failure."""
        logger.info(
            "AdaptiveReflexion: Generating reflection | cycle={}/{}",
            self._state.reflection_cycles + 1,
            self.max_reflections,
        )

        reflection_messages = list(messages) + [
            {"role": "user", "content": _templates.reflection_prompt(failure_context)}
        ]

        response = await self.llm_client.agenerate(
            messages=reflection_messages,
            model=self.model,
            mode="text",
        )

        insight_text = response.content or "Unable to generate insight"

        return ReflectionInsight(
            iteration=self._state.iteration,
            failure_description=failure_context,
            insight=insight_text,
        )

    def _should_decompose(self) -> bool:
        """Determine if we should switch to decomposition."""
        return (
            self._state.reflection_cycles >= self.max_reflections
            and not self._state.decomposition_triggered
        )

    async def plan(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool = True,
    ) -> StrategyOutput:
        """
        Generate next actions using combined ADaPT + Reflexion pattern.
        """
        self._state.iteration += 1
        can_decompose = self._has_sub_agent_tool(tools)

        # Detect failure from previous iteration
        failure_detected, failure_desc = self._detect_failure(messages)

        if failure_detected:
            self._state.consecutive_failures += 1
            self._state.pending_reflection = True
            self._state.last_failure_context = failure_desc

            logger.debug(
                "AdaptiveReflexion: Failure detected | consecutive={} | desc={}",
                self._state.consecutive_failures,
                failure_desc[:50],
            )

        # Phase transitions based on state
        if self._state.phase == AdaptivePhase.DIRECT:
            self._state.direct_attempts += 1

            if self._state.pending_reflection:
                # Transition to reflecting
                self._state.phase = AdaptivePhase.REFLECTING
            elif self._state.direct_attempts > self.max_direct_attempts:
                # Force reflection even without explicit failure
                self._state.phase = AdaptivePhase.REFLECTING
                self._state.last_failure_context = "Max direct attempts without success"

        # Handle reflecting phase
        if self._state.phase == AdaptivePhase.REFLECTING:
            if self._should_decompose() and can_decompose:
                # Switch to decomposition
                logger.info(
                    "AdaptiveReflexion: Reflections exhausted, switching to decomposition"
                )
                self._state.phase = AdaptivePhase.DECOMPOSING
                self._state.decomposition_triggered = True
            else:
                # Generate reflection and retry
                insight = await self._generate_reflection(
                    messages, self._state.last_failure_context
                )
                self._state.memory.add_insight(insight, persist=self.persist_insights)
                self._state.reflection_cycles += 1
                self._state.pending_reflection = False
                self._state.phase = AdaptivePhase.RETRYING

                logger.debug(
                    "AdaptiveReflexion: Insight stored | total={}",
                    len(self._state.memory.task_insights),
                )

        # Execute based on current phase
        if self._state.phase in (AdaptivePhase.DIRECT, AdaptivePhase.RETRYING):
            return await self._execute_with_insights(
                messages, tools, parallel_tool_calls, include_sub_agent=False
            )

        elif self._state.phase == AdaptivePhase.DECOMPOSING:
            result = await self._execute_decomposition(
                messages, tools, parallel_tool_calls
            )
            self._state.phase = AdaptivePhase.EXECUTING_SUBTASKS
            return result

        elif self._state.phase == AdaptivePhase.EXECUTING_SUBTASKS:
            return await self._execute_with_insights(
                messages, tools, parallel_tool_calls, include_sub_agent=True
            )

        # Fallback
        return await self._execute_with_insights(
            messages, tools, parallel_tool_calls, include_sub_agent=False
        )

    async def _execute_with_insights(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool,
        include_sub_agent: bool,
    ) -> StrategyOutput:
        """Execute with accumulated reflection insights."""
        filtered_tools = self._filter_tools(tools, include_sub_agent)
        tool_names = [t.name for t in filtered_tools]

        # Build context with insights
        insight_context = self._state.memory.get_context_prompt()

        action_messages = list(messages)
        if insight_context:
            action_messages.append({"role": "system", "content": insight_context})

        action_messages.append(
            {"role": "user", "content": _templates.action_prompt(tool_names)}
        )

        logger.debug(
            "AdaptiveReflexion: {} phase | insights={} | tools={}",
            self._state.phase.value,
            len(self._state.memory.task_insights),
            len(filtered_tools),
        )

        response = await self.llm_client.agenerate(
            messages=action_messages,
            model=self.model,
            mode="tool_calling",
            tools=filtered_tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        result = self._process_response(response)

        # Reset phase to DIRECT after successful tool selection (for next iteration)
        if not result.finished and self._state.phase == AdaptivePhase.RETRYING:
            self._state.phase = AdaptivePhase.DIRECT

        return result

    async def _execute_decomposition(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool,
    ) -> StrategyOutput:
        """Generate and execute decomposition plan."""
        tool_names = [t.name for t in tools]

        # Include reflection insights in decomposition context
        insight_context = self._state.memory.get_context_prompt()
        failure_summary = self._state.last_failure_context

        decomp_messages = list(messages)
        if insight_context:
            decomp_messages.append({"role": "system", "content": insight_context})

        decomp_messages.append(
            {
                "role": "user",
                "content": _templates.decomposition_prompt(
                    failure_summary=failure_summary,
                    reflection_count=self._state.reflection_cycles,
                    tool_names=tool_names,
                ),
            }
        )

        logger.info(
            "AdaptiveReflexion: DECOMPOSING | reflections={} | insights={}",
            self._state.reflection_cycles,
            len(self._state.memory.task_insights),
        )

        response = await self.llm_client.agenerate(
            messages=decomp_messages,
            model=self.model,
            mode="tool_calling",
            tools=tools,  # Include SubAgentTool
            parallel_tool_calls=parallel_tool_calls,
        )

        return self._process_response(response)

    def _process_response(self, response: t.Any) -> StrategyOutput:
        """Process LLM response into StrategyOutput."""
        if not response.tool_calls:
            return StrategyOutput(
                finished=True,
                success=False,
                result=response.finish_reason or "No tool calls returned by LLM",
            )

        for tc in response.tool_calls:
            if tc.tool_name.upper() == self.finish_tool_name.upper():
                # Reset consecutive failures on successful finish
                if tc.arguments.get("success", True):
                    self._state.consecutive_failures = 0

                return StrategyOutput(
                    finished=True,
                    success=tc.arguments.get("success", True),
                    result=tc.arguments.get("result", "Task completed"),
                )

        return StrategyOutput(tool_calls=response.tool_calls)

    def reset(self, full: bool = False) -> None:
        """
        Reset strategy state.

        Args:
            full: If True, also clear persistent insights
        """
        self._state.reset()
        if full:
            self._state.memory.reset_all()

    def get_insights(self) -> list[ReflectionInsight]:
        """Get accumulated insights for inspection."""
        return self._state.memory.task_insights + self._state.memory.persistent_insights

    def get_state_summary(self) -> dict[str, t.Any]:
        """Get current state summary for debugging."""
        return {
            "phase": self._state.phase.value,
            "iteration": self._state.iteration,
            "direct_attempts": self._state.direct_attempts,
            "reflection_cycles": self._state.reflection_cycles,
            "decomposition_triggered": self._state.decomposition_triggered,
            "consecutive_failures": self._state.consecutive_failures,
            "insights_count": len(self._state.memory.task_insights),
        }

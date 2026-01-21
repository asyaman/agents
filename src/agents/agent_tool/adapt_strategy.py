"""
ADaPT Strategy - Adaptive Planning through Task Decomposition.

ADaPT (Adaptive Planning) follows a "try simple first, decompose on failure" pattern:
1. First attempts direct execution without decomposition
2. On failure/stagnation, decomposes task into subtasks
3. Executes subtasks via SubAgentTool
4. Synthesizes results

Key insight: Only decompose when simpler approaches fail, avoiding unnecessary overhead.

Reference: "ADaPT: As-Needed Decomposition and Planning with Language Models" (2023)
"""

import typing as t
from enum import Enum

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agents.agent_tool.base_strategy import PlanningStrategy, StrategyOutput
from agents.configs import get_agent_tool_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool

# Load templates
_templates = get_agent_tool_template_module("adapt_strategy.jinja")


class AdaptPhase(str, Enum):
    """Current phase of the ADaPT strategy."""

    DIRECT = "direct"  # Attempting direct execution
    DECOMPOSING = "decomposing"  # Analyzing failure and creating subtasks
    EXECUTING = "executing"  # Running decomposed subtasks


class FailureSignal(BaseModel):
    """Detected failure signal from message history."""

    detected: bool = Field(default=False)
    reason: str = Field(default="")
    error_count: int = Field(default=0)
    stagnation_detected: bool = Field(default=False)


class AdaptStrategy(PlanningStrategy):
    """
    ADaPT Strategy: Try simple first, decompose on failure.

    Flow:
        1. DIRECT phase: Execute with all tools EXCEPT SubAgentTool
        2. Detect failure: Errors, stagnation, or no progress
        3. DECOMPOSING phase: LLM analyzes failure and creates decomposition plan
        4. EXECUTING phase: Use SubAgentTool for subtasks

    Configuration:
        - max_direct_attempts: How many iterations to try direct approach
        - error_threshold: Number of tool errors before triggering decomposition
        - stagnation_window: Iterations without progress before decomposition

    Return behavior matches base strategies:
        - No tool_calls → finished=True, success=False
        - FINISH tool called → finished=True, success=<from args>
        - Other tools called → finished=False, tool_calls=[...]
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str | None = None,
        max_direct_attempts: int = 3,
        error_threshold: int = 2,
        stagnation_window: int = 2,
        finish_tool_name: str = "finish",
        sub_agent_tool_name: str = "delegate_subtask",
        decomposition_prompt: str | None = None,
        direct_prompt: str | None = None,
    ):
        """
        Initialize ADaPT Strategy.

        Args:
            llm_client: LLM client for all phases
            model: Optional model override
            max_direct_attempts: Max iterations before considering decomposition
            error_threshold: Tool errors count triggering decomposition
            stagnation_window: Repeated similar states triggering decomposition
            finish_tool_name: Name of the finish tool
            sub_agent_tool_name: Name of the SubAgentTool for decomposition
            decomposition_prompt: Custom prompt for decomposition phase
            direct_prompt: Custom prompt for direct execution phase
        """
        self.llm_client = llm_client
        self.model = model
        self.max_direct_attempts = max_direct_attempts
        self.error_threshold = error_threshold
        self.stagnation_window = stagnation_window
        self.finish_tool_name = finish_tool_name
        self.sub_agent_tool_name = sub_agent_tool_name
        self.decomposition_prompt = decomposition_prompt
        self.direct_prompt = direct_prompt

        # State tracking
        self._phase = AdaptPhase.DIRECT
        self._direct_attempts = 0
        self._recent_tool_results: list[str] = []
        self._decomposition_triggered = False

    def _get_direct_prompt(self, tool_names: list[str]) -> str:
        """Get prompt for direct execution phase."""
        if self.direct_prompt:
            return self.direct_prompt
        return _templates.direct_prompt(tool_names=tool_names)

    def _get_decomposition_prompt(
        self, failure_reason: str, tool_names: list[str]
    ) -> str:
        """Get prompt for decomposition phase."""
        if self.decomposition_prompt:
            return self.decomposition_prompt
        return _templates.decomposition_prompt(
            failure_reason=failure_reason,
            tool_names=tool_names,
        )

    def _detect_failure(
        self, messages: list[ChatCompletionMessageParam]
    ) -> FailureSignal:
        """
        Analyze message history for failure signals.

        Detects:
        - Tool execution errors
        - Stagnation (repeated similar outputs)
        - Explicit failure indicators
        """
        error_count = 0
        recent_results: list[str] = []

        for msg in messages:
            if msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                recent_results.append(content[:100])  # Track for stagnation

                # Check for error patterns
                if '"error"' in content.lower() or "error:" in content.lower():
                    error_count += 1

        # Update tracking
        self._recent_tool_results = recent_results[-self.stagnation_window * 2 :]

        # Detect stagnation (similar results repeating)
        stagnation = False
        if len(recent_results) >= self.stagnation_window:
            window = recent_results[-self.stagnation_window :]
            if len(set(window)) == 1 and window[0]:  # All same non-empty result
                stagnation = True

        # Build failure signal
        failure_detected = (
            error_count >= self.error_threshold
            or stagnation
            or self._direct_attempts >= self.max_direct_attempts
        )

        reason_parts = []
        if error_count >= self.error_threshold:
            reason_parts.append(f"{error_count} tool errors detected")
        if stagnation:
            reason_parts.append("execution appears stagnated")
        if self._direct_attempts >= self.max_direct_attempts:
            reason_parts.append(
                f"max direct attempts ({self.max_direct_attempts}) reached"
            )

        return FailureSignal(
            detected=failure_detected,
            reason="; ".join(reason_parts) if reason_parts else "",
            error_count=error_count,
            stagnation_detected=stagnation,
        )

    def _filter_tools_for_phase(
        self, tools: list[BaseTool[t.Any, t.Any]], phase: AdaptPhase
    ) -> list[BaseTool[t.Any, t.Any]]:
        """Filter tools based on current phase."""
        if phase == AdaptPhase.DIRECT:
            # Exclude SubAgentTool during direct phase
            return [
                t for t in tools if t.name.lower() != self.sub_agent_tool_name.lower()
            ]
        else:
            # Include all tools during decomposition/execution
            return list(tools)

    def _has_sub_agent_tool(self, tools: list[BaseTool[t.Any, t.Any]]) -> bool:
        """Check if SubAgentTool is available."""
        return any(t.name.lower() == self.sub_agent_tool_name.lower() for t in tools)

    async def plan(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool = True,
    ) -> StrategyOutput:
        """
        Generate next actions using ADaPT pattern.

        Phase transitions:
            DIRECT → (failure detected) → DECOMPOSING → EXECUTING
            DIRECT → (success) → finish
            EXECUTING → (subtasks complete) → finish
        """
        # Check if decomposition is even possible
        can_decompose = self._has_sub_agent_tool(tools)

        # Detect failure signals
        failure = self._detect_failure(messages)

        # Phase transition logic
        if self._phase == AdaptPhase.DIRECT:
            self._direct_attempts += 1

            if failure.detected and can_decompose and not self._decomposition_triggered:
                logger.info(
                    "ADaPT: Failure detected, switching to decomposition | reason={}",
                    failure.reason,
                )
                self._phase = AdaptPhase.DECOMPOSING
                self._decomposition_triggered = True
            else:
                # Continue direct execution
                return await self._execute_direct(messages, tools, parallel_tool_calls)

        if self._phase == AdaptPhase.DECOMPOSING:
            # Generate decomposition plan and switch to executing
            result = await self._execute_decomposition(
                messages, tools, failure.reason, parallel_tool_calls
            )
            self._phase = AdaptPhase.EXECUTING
            return result

        # EXECUTING phase - use all tools including SubAgentTool
        return await self._execute_with_subtasks(messages, tools, parallel_tool_calls)

    async def _execute_direct(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool,
    ) -> StrategyOutput:
        """Execute direct approach without decomposition."""
        filtered_tools = self._filter_tools_for_phase(tools, AdaptPhase.DIRECT)
        tool_names = [t.name for t in filtered_tools]

        logger.debug(
            "ADaPT DIRECT phase | attempt={}/{} | tools={}",
            self._direct_attempts,
            self.max_direct_attempts,
            len(filtered_tools),
        )

        action_messages = list(messages) + [
            {"role": "user", "content": self._get_direct_prompt(tool_names)}
        ]

        response = await self.llm_client.agenerate(
            messages=action_messages,
            model=self.model,
            mode="tool_calling",
            tools=filtered_tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        return self._process_response(response)

    async def _execute_decomposition(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        failure_reason: str,
        parallel_tool_calls: bool,
    ) -> StrategyOutput:
        """Generate decomposition plan and delegate to subtasks."""
        all_tools = self._filter_tools_for_phase(tools, AdaptPhase.EXECUTING)
        tool_names = [t.name for t in all_tools]

        logger.info("ADaPT DECOMPOSING phase | reason={}", failure_reason)

        decomp_messages = list(messages) + [
            {
                "role": "user",
                "content": self._get_decomposition_prompt(failure_reason, tool_names),
            }
        ]

        response = await self.llm_client.agenerate(
            messages=decomp_messages,
            model=self.model,
            mode="tool_calling",
            tools=all_tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        return self._process_response(response)

    async def _execute_with_subtasks(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool,
    ) -> StrategyOutput:
        """Execute with full tool access including SubAgentTool."""
        all_tools = self._filter_tools_for_phase(tools, AdaptPhase.EXECUTING)
        tool_names = [t.name for t in all_tools]

        logger.debug("ADaPT EXECUTING phase | tools={}", len(all_tools))

        action_messages = list(messages) + [
            {"role": "user", "content": self._get_direct_prompt(tool_names)}
        ]

        response = await self.llm_client.agenerate(
            messages=action_messages,
            model=self.model,
            mode="tool_calling",
            tools=all_tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        return self._process_response(response)

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

    def reset(self) -> None:
        """Reset strategy state for new task."""
        self._phase = AdaptPhase.DIRECT
        self._direct_attempts = 0
        self._recent_tool_results = []
        self._decomposition_triggered = False

"""
SubAgentTool - A tool that delegates sub-objectives to independent agent instances.

This tool enables recursive agent execution where:
- Each invocation creates a FRESH AgentTool with independent message history
- Parent agents only see the result/success (opaque output)
- Full execution details are collected in the global execution history
- Tool selection can filter relevant tools for each sub-objective
"""

import typing as t
from collections.abc import Callable
import asyncio

from loguru import logger
from pydantic import BaseModel, Field

from agents.agent_tool.agent_tool import AgentTool, AgentToolInput
from agents.agent_tool.base_strategy import PlanningStrategy
from agents.configs import get_agent_tool_template_module
from agents.agent_tool.recursive.context import (
    LevelExecution,
    can_recurse,
    get_current_depth,
    get_max_depth,
    increment_depth,
    record_level_execution,
    var_parent_objective,
    var_recursion_depth,
)
from agents.tools_core.base_tool import BaseTool
from agents.tools_core.internal_tools.tool_selector import ToolSelector


class SubAgentInput(BaseModel):
    """Input for the SubAgentTool."""

    sub_objective: str = Field(
        description="The sub-task or sub-objective to accomplish. Be specific and clear."
    )
    context: str | None = Field(
        default=None,
        description="Additional context to help accomplish the sub-objective.",
    )


class SubAgentOutput(BaseModel):
    """
    Output from the SubAgentTool.

    Intentionally opaque - parent only sees result and success status,
    not the internal message history or execution details.
    """

    result: str = Field(description="The result of the sub-task.")
    success: bool = Field(
        description="Whether the sub-task was completed successfully."
    )


class SubAgentTool(BaseTool[SubAgentInput, SubAgentOutput]):
    """
    A tool that delegates sub-objectives to independent agent instances.

    When invoked, this tool:
    1. Checks if further recursion is allowed (depth limit)
    2. Selects relevant tools for the sub-objective (if tool_selector provided)
    3. Creates a FRESH AgentTool with independent message history
    4. Executes the sub-objective
    5. Records execution to global history
    6. Returns only result/success (isolation from parent)

    The parent agent sees this as a regular tool with simple I/O,
    unaware of the full execution that happened internally.

    Usage:
        sub_agent = SubAgentTool(
            available_tools=[search_tool, calc_tool],
            strategy_factory=lambda: DirectStrategy(llm_client),
            tool_selector=ToolSelector(llm_client),  # Optional
        )

        # Can be passed to any AgentTool
        agent = AgentTool(
            tools=[other_tool, sub_agent],
            strategy=DirectStrategy(llm_client),
        )
    """

    _name = "delegate_subtask"
    description = (
        "Delegate a complex sub-task to a specialized agent. Use this when a task "
        "can be broken down into independent sub-objectives that require their own "
        "planning and execution. The sub-agent will have access to appropriate tools "
        "and will return the result when complete."
    )
    _input = SubAgentInput
    _output = SubAgentOutput

    example_inputs: t.ClassVar[t.Sequence[SubAgentInput]] = (
        SubAgentInput(
            sub_objective="Research the latest AI developments in 2024",
            context="Focus on language models and their applications",
        ),
        SubAgentInput(
            sub_objective="Calculate the compound interest for the investment",
            context="Principal: $10000, Rate: 5%, Time: 3 years",
        ),
    )

    example_outputs: t.ClassVar[t.Sequence[SubAgentOutput]] = (
        SubAgentOutput(
            result="Found 5 major developments: 1) GPT-4 improvements...",
            success=True,
        ),
        SubAgentOutput(
            result="Compound interest is $1576.25, final amount is $11576.25",
            success=True,
        ),
    )

    @classmethod
    def get_guidance_prompt(cls) -> str:
        """
        Get the guidance prompt explaining when and how to use delegate_subtask.

        This prompt is intended to be injected as a system message when
        the SubAgentTool is available to an agent.

        Returns:
            Formatted guidance text for the LLM.
        """
        templates = get_agent_tool_template_module("sub_agent_tool.jinja")
        return templates.guidance()

    def __init__(
        self,
        available_tools: list[BaseTool[t.Any, t.Any]],
        strategy_factory: Callable[[], PlanningStrategy],
        tool_selector: ToolSelector | None = None,
        max_iterations_per_level: int = 10,
        include_self_in_children: bool = True,
        system_prompt: str | None = None,
        parallel_tool_calls: bool = True,
    ) -> None:
        """
        Initialize SubAgentTool.

        Args:
            available_tools: Tools available for child agents (excluding self)
            strategy_factory: Factory to create fresh PlanningStrategy per invocation
            tool_selector: Optional ToolSelector to filter tools per sub-objective
            max_iterations_per_level: Max iterations for each child agent
            include_self_in_children: Whether child agents can recurse further
            system_prompt: Custom system prompt for child agents
            parallel_tool_calls: Allow parallel tool calls in child agents
        """
        super().__init__()
        self.available_tools = list(available_tools)
        self.strategy_factory = strategy_factory
        self.tool_selector = tool_selector
        self.max_iterations_per_level = max_iterations_per_level
        self.include_self_in_children = include_self_in_children
        self.system_prompt = system_prompt
        self.parallel_tool_calls = parallel_tool_calls

    def invoke(self, input: SubAgentInput) -> SubAgentOutput:
        """Sync execution - not recommended for recursive agents."""

        return asyncio.run(self.ainvoke(input))

    async def ainvoke(self, input: SubAgentInput) -> SubAgentOutput:
        """Execute sub-objective in an independent agent instance."""
        validated = self._validate_input(input)
        current_depth = get_current_depth()
        max_depth = get_max_depth()

        logger.info(
            "SubAgentTool invoked | depth={}/{} | objective={}",
            current_depth,
            max_depth,
            (
                validated.sub_objective[:50] + "..."
                if len(validated.sub_objective) > 50
                else validated.sub_objective
            ),
        )

        # Check depth limit
        if not can_recurse():
            logger.warning(
                "Max recursion depth reached | depth={} | max={}",
                current_depth,
                max_depth,
            )
            return SubAgentOutput(
                result=f"Cannot process: maximum recursion depth ({max_depth}) reached",
                success=False,
            )

        # Select tools for this sub-objective
        selected_tools = await self._select_tools(validated.sub_objective)

        # Optionally add self for deeper recursion
        if self.include_self_in_children and current_depth < max_depth - 1:
            selected_tools.append(self)
            logger.debug(
                "Added self to child tools | child_depth={}",
                current_depth + 1,
            )

        # Create fresh agent (independent message history)
        child_agent = AgentTool(
            tools=selected_tools,
            strategy=self.strategy_factory(),  # Fresh strategy instance!
            system_prompt=self.system_prompt,
            include_finish_tool=True,
            parallel_tool_calls=self.parallel_tool_calls,
        )

        # Execute in child context (increment depth)
        parent_objective = var_parent_objective.get("")
        var_parent_objective.set(validated.sub_objective)
        child_depth = increment_depth()

        logger.debug(
            "Executing child agent | depth={} | tools={}",
            child_depth,
            [t.name for t in selected_tools],
        )

        try:
            result = await child_agent.ainvoke(
                AgentToolInput(
                    objective=validated.sub_objective,
                    context=validated.context,
                    max_iterations=self.max_iterations_per_level,
                )
            )
        finally:
            # Restore parent context
            var_recursion_depth.set(current_depth)
            var_parent_objective.set(parent_objective)

        # Count tool calls from messages
        tool_calls_count = sum(
            1
            for msg in result.messages
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        )

        # Record to global history
        record_level_execution(
            LevelExecution(
                depth=child_depth,
                objective=validated.sub_objective,
                iterations_used=result.iterations_used,
                success=result.success,
                result=result.result,
                messages=result.messages,
                tool_calls_count=tool_calls_count,
            )
        )

        logger.info(
            "Child agent completed | depth={} | success={} | iterations={}",
            child_depth,
            result.success,
            result.iterations_used,
        )

        # Return opaque result (parent doesn't see internal messages)
        return SubAgentOutput(
            result=result.result,
            success=result.success,
        )

    async def _select_tools(self, objective: str) -> list[BaseTool[t.Any, t.Any]]:
        """Select relevant tools for the given objective."""
        if self.tool_selector is None:
            return list(self.available_tools)

        logger.debug(
            "Selecting tools for objective | available={}",
            len(self.available_tools),
        )

        selected = await self.tool_selector.afilter_tools(
            objective=objective,
            tools=self.available_tools,
        )

        logger.debug(
            "Tool selection complete | selected={} | tools={}",
            len(selected),
            [t.name for t in selected],
        )

        return selected

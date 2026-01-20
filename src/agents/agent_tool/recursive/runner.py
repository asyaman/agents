"""
RecursiveAgentRunner - Entry point for recursive agent execution.

Provides a clean interface to run recursive agent tasks with:
- Automatic context initialization
- Root agent setup with SubAgentTool
- Complete execution history collection
- Aggregated statistics
"""

import typing as t
from collections.abc import Callable
import asyncio

from loguru import logger
from pydantic import BaseModel, Field

from agents.agent_tool.agent_tool import AgentTool, AgentToolInput
from agents.agent_tool.planning_strategies import PlanningStrategy
from agents.agent_tool.recursive.context import (
    LevelExecution,
    RecursionStatistics,
    initialize_recursion_context,
)
from agents.agent_tool.recursive.sub_agent_tool import SubAgentTool
from agents.tools_core.base_tool import BaseTool
from agents.tools_core.internal_tools.tool_selector import ToolSelector


class RecursiveAgentOutput(BaseModel):
    """
    Complete output from a recursive agent execution.

    Contains the final result plus full execution history
    from all recursion levels for debugging and analysis.
    """

    model_config = {"arbitrary_types_allowed": True}

    result: str = Field(description="The final result of the task.")
    success: bool = Field(description="Whether the task completed successfully.")
    iterations_used: int = Field(description="Iterations used at root level.")
    statistics: RecursionStatistics = Field(
        description="Aggregated statistics across all recursion levels."
    )
    execution_history: list[LevelExecution] = Field(
        default_factory=list,
        description="Execution records from all levels (root first, then by execution order).",
    )


class RecursiveAgentRunner:
    """
    Entry point for running recursive agent tasks.

    This runner:
    1. Initializes context variables for tracking
    2. Creates a SubAgentTool for recursive delegation
    3. Creates root AgentTool with the SubAgentTool included
    4. Executes the task
    5. Collects and returns complete execution history

    Usage:
        runner = RecursiveAgentRunner(
            tools=[search_tool, calculator_tool],
            strategy_factory=lambda: DirectStrategy(llm_client),
            tool_selector=ToolSelector(llm_client),  # Optional
            max_depth=3,
        )

        result = await runner.run(
            objective="Research and summarize AI trends",
            context="Focus on 2024 developments"
        )

        # Access detailed history
        for level in result.execution_history:
            print(f"Depth {level.depth}: {level.objective}")
            print(f"  Success: {level.success}, Iterations: {level.iterations_used}")
    """

    def __init__(
        self,
        tools: list[BaseTool[t.Any, t.Any]],
        strategy_factory: Callable[[], PlanningStrategy],
        tool_selector: ToolSelector | None = None,
        max_depth: int = 3,
        max_iterations_per_level: int = 5,
        system_prompt: str | None = None,
        parallel_tool_calls: bool = True,
        include_sub_agent_at_root: bool = True,
    ) -> None:
        """
        Initialize RecursiveAgentRunner.

        Args:
            tools: Base tools available to all agents
            strategy_factory: Factory to create fresh PlanningStrategy per agent
            tool_selector: Optional ToolSelector to filter tools per objective
            max_depth: Maximum recursion depth (root = 0)
            max_iterations_per_level: Max iterations per agent execution
            system_prompt: Custom system prompt for agents
            parallel_tool_calls: Allow parallel tool calls
            include_sub_agent_at_root: Whether root agent has SubAgentTool
        """
        self.tools = list(tools)
        self.strategy_factory = strategy_factory
        self.tool_selector = tool_selector
        self.max_depth = max_depth
        self.max_iterations_per_level = max_iterations_per_level
        self.system_prompt = system_prompt
        self.parallel_tool_calls = parallel_tool_calls
        self.include_sub_agent_at_root = include_sub_agent_at_root

    async def run(
        self,
        objective: str,
        context: str | None = None,
    ) -> RecursiveAgentOutput:
        """
        Run a recursive agent task.

        Args:
            objective: The main objective to accomplish
            context: Optional additional context

        Returns:
            RecursiveAgentOutput with result, statistics, and full execution history
        """
        logger.info(
            "Starting recursive agent | objective={} | max_depth={} | tools={}",
            objective[:50] + "..." if len(objective) > 50 else objective,
            self.max_depth,
            len(self.tools),
        )

        # Initialize context variables
        execution_history, statistics = initialize_recursion_context(
            max_depth=self.max_depth
        )

        # Build root tools and get guidance if sub-agent is included
        root_tools, guidance_messages = await self._build_root_tools(objective)

        # Create root agent
        root_agent = AgentTool(
            tools=root_tools,
            strategy=self.strategy_factory(),
            system_prompt=self.system_prompt,
            include_finish_tool=True,
            parallel_tool_calls=self.parallel_tool_calls,
            guidance_messages=guidance_messages,
        )

        logger.debug(
            "Root agent created | tools={}",
            [t.name for t in root_tools],
        )

        # Execute root agent
        result = await root_agent.ainvoke(
            AgentToolInput(
                objective=objective,
                context=context,
                max_iterations=self.max_iterations_per_level,
            )
        )

        # Count tool calls at root level
        tool_calls_count = sum(
            1
            for msg in result.messages
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        )

        # Record root level execution at the beginning
        root_execution = LevelExecution(
            depth=0,
            objective=objective,
            iterations_used=result.iterations_used,
            success=result.success,
            result=result.result,
            messages=result.messages,
            tool_calls_count=tool_calls_count,
        )

        # Update statistics for root level
        statistics.update_from_level(
            depth=0,
            iterations=result.iterations_used,
            tool_calls=tool_calls_count,
        )

        # Insert root at beginning (execution_history contains child executions)
        all_history = [root_execution] + execution_history

        logger.info(
            "Recursive agent completed | success={} | levels={} | total_iterations={}",
            result.success,
            statistics.levels_completed,
            statistics.total_iterations,
        )

        return RecursiveAgentOutput(
            result=result.result,
            success=result.success,
            iterations_used=result.iterations_used,
            statistics=statistics,
            execution_history=all_history,
        )

    async def _build_root_tools(
        self, objective: str
    ) -> tuple[list[BaseTool[t.Any, t.Any]], list[str]]:
        """Build the tool list and guidance messages for the root agent.

        Returns:
            Tuple of (tools, guidance_messages)
        """
        guidance_messages: list[str] = []

        # Start with base tools (optionally filtered)
        if self.tool_selector:
            selected_tools = await self.tool_selector.afilter_tools(
                objective=objective,
                tools=self.tools,
            )
            logger.debug(
                "Root tool selection | selected={}/{}",
                len(selected_tools),
                len(self.tools),
            )
        else:
            selected_tools = list(self.tools)

        # Add SubAgentTool if enabled and depth allows
        if self.include_sub_agent_at_root and self.max_depth > 1:
            sub_agent = SubAgentTool(
                available_tools=self.tools,  # Children get access to all tools
                strategy_factory=self.strategy_factory,
                tool_selector=self.tool_selector,
                max_iterations_per_level=self.max_iterations_per_level,
                include_self_in_children=True,
                system_prompt=self.system_prompt,
                parallel_tool_calls=self.parallel_tool_calls,
            )
            selected_tools.append(sub_agent)
            guidance_messages.append(SubAgentTool.get_guidance_prompt())
            logger.debug("Added SubAgentTool to root agent with guidance")

        return selected_tools, guidance_messages

    def run_sync(
        self,
        objective: str,
        context: str | None = None,
    ) -> RecursiveAgentOutput:
        """Synchronous wrapper for run()."""

        return asyncio.run(self.run(objective, context))


# Convenience function for quick usage
async def run_recursive_agent(
    objective: str,
    tools: list[BaseTool[t.Any, t.Any]],
    strategy_factory: Callable[[], PlanningStrategy],
    context: str | None = None,
    tool_selector: "ToolSelector | None" = None,
    max_depth: int = 3,
    max_iterations_per_level: int = 10,
) -> RecursiveAgentOutput:
    """
    Convenience function to run a recursive agent task.

    Args:
        objective: The main objective to accomplish
        tools: Tools available to agents
        strategy_factory: Factory to create planning strategies
        context: Optional additional context
        tool_selector: Optional tool selector for filtering
        max_depth: Maximum recursion depth
        max_iterations_per_level: Max iterations per level

    Returns:
        RecursiveAgentOutput with complete results
    """
    runner = RecursiveAgentRunner(
        tools=tools,
        strategy_factory=strategy_factory,
        tool_selector=tool_selector,
        max_depth=max_depth,
        max_iterations_per_level=max_iterations_per_level,
    )
    return await runner.run(objective, context)

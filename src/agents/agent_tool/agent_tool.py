"""
AgentTool - General-purpose agent executor with pluggable planning strategies.

AgentTool handles:
- Message history management (list[ChatCompletionMessageParam])
- Tool execution
- Iteration control

The planning strategy handles:
- LLM interaction
- Reasoning (if applicable)
- Tool selection
"""

import asyncio
import typing as t
import json

from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agents.configs import get_agent_tool_template_module
from agents.tools_core.base_tool import BaseTool
from agents.agent_tool.planning_strategies import PlanningStrategy
from agents.llm_core.llm_client import ToolCall
from agents.tools_core.base_tool import create_fn_tool

# Load templates
_templates = get_agent_tool_template_module("agent_tool.jinja")


class AgentToolInput(BaseModel):
    """Input for the AgentTool."""

    objective: str = Field(description="The task/objective to accomplish.")
    context: str | None = Field(
        default=None,
        description="Additional context to help accomplish the objective.",
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations before stopping.",
    )


class AgentToolOutput(BaseModel):
    """Output from the AgentTool."""

    result: str = Field(description="The result of the task.")
    success: bool = Field(description="Whether the task was completed successfully.")
    iterations_used: int = Field(description="Number of iterations used.")
    # Note: Using dict instead of ChatCompletionMessageParam to avoid Pydantic v2
    # serialization issues with OpenAI's complex Union types
    messages: list[dict[str, t.Any]] = Field(
        default_factory=list,
        description="Full conversation history (for debugging/inspection).",
    )


class FinishInput(BaseModel):
    """Input for the finish tool."""

    result: str = Field(description="The final result/answer for the objective.")
    success: bool = Field(default=True, description="Whether the task was successful.")


class FinishOutput(BaseModel):
    """Output from the finish tool."""

    acknowledged: bool = Field(default=True)


def create_finish_tool() -> BaseTool[FinishInput, FinishOutput]:
    """Create the finish tool that signals task completion."""

    @create_fn_tool(
        name="finish",
        description="Call this when the task is complete to return the final result.",
    )
    def finish(result: str, success: bool = True) -> FinishOutput:  # noqa: ARG001
        return FinishOutput(acknowledged=True)

    return finish  # type: ignore


class AgentTool(BaseTool[AgentToolInput, AgentToolOutput]):
    """
    General-purpose agent tool with pluggable planning strategy.

    The AgentTool provides an agentic loop where:
    1. Strategy decides what to do (reasoning + tool selection)
    2. AgentTool executes the tools
    3. Results are added to message history
    4. Repeat until task is complete or max iterations

    Usage:
        # Simple direct execution
        agent = AgentTool(
            tools=[search_tool, calculator_tool],
            llm_client=client,
            strategy=DirectStrategy(),
        )

        # With reasoning (React pattern)
        agent = AgentTool(
            tools=[search_tool, calculator_tool],
            llm_client=client,
            strategy=ReactStrategy(),
        )

        # Execute
        result = await agent.ainvoke(AgentToolInput(
            objective="Find the population of France and calculate 10% of it"
        ))
    """

    _name = "agent"
    description = "A general-purpose agent that can use tools to accomplish objectives."
    _input = AgentToolInput
    _output = AgentToolOutput

    def __init__(
        self,
        tools: list[BaseTool[t.Any, t.Any]],
        strategy: PlanningStrategy,
        system_prompt: str | None = None,
        include_finish_tool: bool = True,
        parallel_tool_calls: bool = True,
        guidance_messages: list[str] | None = None,
    ) -> None:
        """
        Initialize AgentTool.

        Args:
            tools: List of tools available to the agent
            strategy: Planning strategy (owns its LLM client and model config)
            system_prompt: Custom system prompt (uses template if None)
            include_finish_tool: Whether to auto-add finish tool (default True)
            parallel_tool_calls: Allow parallel tool calls (LLM and execution)
            guidance_messages: Additional system messages to inject after the main
                system prompt (e.g., sub-agent usage guidance)
        """
        super().__init__()
        self.strategy = strategy
        self._system_prompt = system_prompt
        self.parallel_tool_calls = parallel_tool_calls
        self.guidance_messages = guidance_messages or []

        # Add finish tool if requested
        self.tools = list(tools)
        if include_finish_tool:
            self.tools.append(create_finish_tool())

    def _get_system_prompt(self, tools: list[BaseTool[t.Any, t.Any]]) -> str:
        """Get the system prompt from template or custom."""
        if self._system_prompt:
            return self._system_prompt
        return _templates.system_prompt(tools=tools)

    def _get_task_prompt(self, objective: str, context: str | None) -> str:
        """Get the task prompt from template."""
        return _templates.task_prompt(objective=objective, context=context)

    def invoke(self, input: AgentToolInput) -> AgentToolOutput:
        """Sync execution - wraps async implementation."""
        return asyncio.run(self.ainvoke(input))

    async def ainvoke(self, input: AgentToolInput) -> AgentToolOutput:
        """Execute the agent task asynchronously."""
        validated = self._validate_input(input)

        logger.info(
            "Starting agent task | objective={} | max_iterations={} | strategy={}",
            (
                validated.objective[:50] + "..."
                if len(validated.objective) > 50
                else validated.objective
            ),
            validated.max_iterations,
            type(self.strategy).__name__,
        )

        result = await self._agent_loop(
            objective=validated.objective,
            context=validated.context,
            max_iterations=validated.max_iterations,
        )

        if result.success:
            logger.success(
                "Task completed | iterations={}",
                result.iterations_used,
            )
        else:
            logger.warning(
                "Task failed | iterations={} | result={}",
                result.iterations_used,
                result.result[:100] if result.result else "No result",
            )

        return result

    async def _agent_loop(
        self,
        objective: str,
        context: str | None,
        max_iterations: int,
    ) -> AgentToolOutput:
        """Run the agent loop until task completion or max iterations."""

        # Initialize messages with system prompt and task
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._get_system_prompt(self.tools)},
        ]

        # Inject guidance messages (e.g., sub-agent usage hints)
        for guidance in self.guidance_messages:
            messages.append({"role": "system", "content": guidance})

        messages.append(
            {"role": "user", "content": self._get_task_prompt(objective, context)}
        )

        # Use uppercase keys for case-insensitive lookup (tool.name is normalized to uppercase)
        tool_map = {tool.name.upper(): tool for tool in self.tools}

        for iteration in range(max_iterations):
            logger.debug(
                "Agent iteration {}/{} | messages={}",
                iteration + 1,
                max_iterations,
                len(messages),
            )

            # Delegate to strategy for planning
            strategy_output = await self.strategy.plan(
                messages=messages,
                tools=self.tools,
                parallel_tool_calls=self.parallel_tool_calls,
            )

            # Add any messages generated by strategy (e.g., reasoning)
            messages.extend(strategy_output.messages)

            # Check if task is finished
            if strategy_output.finished:
                return AgentToolOutput(
                    result=strategy_output.result or "",
                    success=strategy_output.success,
                    iterations_used=iteration + 1,
                    messages=t.cast(list[dict[str, t.Any]], messages),
                )

            # Execute tool calls
            if strategy_output.tool_calls:
                await self._execute_tool_calls(
                    tool_calls=strategy_output.tool_calls,
                    tool_map=tool_map,
                    messages=messages,
                    parallel=self.parallel_tool_calls,
                )

        # Max iterations reached
        logger.warning(
            "Max iterations reached | max={}",
            max_iterations,
        )
        return AgentToolOutput(
            result="Max iterations reached without completing the task",
            success=False,
            iterations_used=max_iterations,
            messages=t.cast(list[dict[str, t.Any]], messages),
        )

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        tool_map: dict[str, BaseTool[t.Any, t.Any]],
        messages: list[ChatCompletionMessageParam],
        parallel: bool = True,
    ) -> None:
        """Execute tool calls and add results to messages.

        Args:
            tool_calls: List of tool calls to execute
            tool_map: Mapping of tool names to tool instances
            messages: Message history to append to
            parallel: If True, execute tools in parallel; otherwise sequential
        """

        # Add single assistant message with ALL tool calls
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": tc.parsed.model_dump_json(),
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        async def execute_single_tool(tool_call: ToolCall) -> tuple[str, str]:
            """Execute a single tool and return (tool_call_id, result)."""
            tool_name = tool_call.tool_name
            tool_input = tool_call.parsed

            logger.info(
                "Tool call | tool={} | args={}",
                tool_name,
                tool_input.model_dump_json()[:100],
            )

            tool = tool_map.get(tool_name.upper())
            if tool is None:
                tool_result = json.dumps({"error": f"Unknown tool '{tool_name}'"})
                logger.error("Unknown tool: {}", tool_name)
            else:
                try:
                    result = await tool.acall(tool_input)
                    tool_result = result.model_dump_json()
                    logger.debug("Tool result: {}", tool_result[:200])
                except Exception as e:
                    tool_result = json.dumps(
                        {"error": f"Error executing {tool_name}: {e}"}
                    )
                    logger.error("Tool execution error: {}", e)

            return (tool_call.id, tool_result)

        # Execute tools in parallel or sequentially
        if parallel:
            results = await asyncio.gather(
                *[execute_single_tool(tc) for tc in tool_calls]
            )
        else:
            results = [await execute_single_tool(tc) for tc in tool_calls]

        # Add tool result messages (order matches tool_calls order)
        for tool_call_id, tool_result in results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result,
                }
            )

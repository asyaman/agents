"""
Chainlit instrumentation for AgentTool.

Provides wrappers that add Chainlit UI elements (steps, task list)
to AgentTool execution without modifying the core agent logic.
"""

import typing as t
from copy import deepcopy

import chainlit as cl
from loguru import logger
from pydantic import BaseModel

from agents.agent_tool.agent_tool import AgentTool, AgentToolInput, AgentToolOutput
from agents.agent_tool.base_strategy import StrategyOutput
from agents.llm_core.llm_client import ToolCall
from agents.tools_core.base_tool import BaseTool


def convert_for_chainlit(data: t.Any) -> str | dict[str, t.Any]:
    """Convert data types to Chainlit-compatible format (str or dict)."""
    if isinstance(data, dict):
        return data
    if isinstance(data, BaseModel):
        return data.model_dump()
    return str(data)


def safe_copy(obj: t.Any) -> t.Any:
    """Safely copy objects, falling back to original if copy fails."""
    try:
        return deepcopy(obj)
    except Exception:
        return obj


class ChainlitInstrumentedAgent(AgentTool):
    """
    AgentTool wrapper that adds Chainlit UI instrumentation.

    Shows:
    - Reasoning steps (from ReactStrategy)
    - Tool call steps with input/output
    - Task list progress
    """

    def __init__(
        self,
        agent: AgentTool,
        show_reasoning: bool = True,
        show_tool_calls: bool = True,
        show_task_list: bool = True,
    ):
        """
        Wrap an AgentTool with Chainlit instrumentation.

        Args:
            agent: The AgentTool to wrap
            show_reasoning: Show reasoning/thinking steps
            show_tool_calls: Show each tool call as a step
            show_task_list: Show Chainlit task list
        """
        # Copy essential attributes from wrapped agent
        self.wrapped_agent = agent
        self.tools = agent.tools
        self.strategy = agent.strategy
        self.parallel_tool_calls = agent.parallel_tool_calls
        self.guidance_messages = agent.guidance_messages

        self.show_reasoning = show_reasoning
        self.show_tool_calls = show_tool_calls
        self.show_task_list = show_task_list

        # Task list for progress tracking
        self._task_list: cl.TaskList | None = None
        self._iteration_task: cl.Task | None = None

    async def ainvoke(self, input: AgentToolInput) -> AgentToolOutput:
        """Execute agent with Chainlit instrumentation."""
        # Validate input at entry point (consistent with AgentTool pattern)
        validated = self.wrapped_agent._validate_input(input)

        # Initialize task list
        if self.show_task_list:
            self._task_list = cl.TaskList()
            self._task_list.status = "Running..."
            await self._task_list.send()

        try:
            # Run agent loop with instrumentation
            async with cl.Step(name="Agent", type="run") as main_step:
                main_step.input = validated.objective

                result = await self._instrumented_loop(validated)

                main_step.output = result.result

            # Update task list status
            if self._task_list:
                self._task_list.status = "Complete" if result.success else "Failed"
                await self._task_list.send()

            return result

        except Exception:
            if self._task_list:
                self._task_list.status = "Error"
                await self._task_list.send()
            raise

    async def _instrumented_loop(self, validated: AgentToolInput) -> AgentToolOutput:
        """Run the agent loop with step instrumentation."""
        from openai.types.chat import ChatCompletionMessageParam

        # Initialize messages
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": self.wrapped_agent._get_system_prompt(self.tools),
            },
        ]

        for guidance in self.guidance_messages:
            messages.append({"role": "system", "content": guidance})

        messages.append(
            {
                "role": "user",
                "content": self.wrapped_agent._get_task_prompt(
                    validated.objective, validated.context
                ),
            }
        )

        tool_map = {tool.name.upper(): tool for tool in self.tools}

        for iteration in range(validated.max_iterations):
            # Update iteration in task list
            if self._task_list:
                if self._iteration_task:
                    self._iteration_task.status = cl.TaskStatus.DONE
                self._iteration_task = cl.Task(
                    title=f"Iteration {iteration + 1}",
                    status=cl.TaskStatus.RUNNING,
                )
                await self._task_list.add_task(self._iteration_task)
                await self._task_list.send()

            # Plan phase with instrumentation
            strategy_output = await self._instrumented_plan(messages, iteration)

            # Add strategy messages to history
            messages.extend(strategy_output.messages)

            # Check if finished
            if strategy_output.finished:
                if self._iteration_task:
                    self._iteration_task.status = cl.TaskStatus.DONE
                    await self._task_list.send() if self._task_list else None

                return AgentToolOutput(
                    result=strategy_output.result or "",
                    success=strategy_output.success,
                    iterations_used=iteration + 1,
                    messages=t.cast(list[dict[str, t.Any]], messages),
                )

            # Execute tool calls with instrumentation
            if strategy_output.tool_calls:
                await self._instrumented_tool_execution(
                    tool_calls=strategy_output.tool_calls,
                    tool_map=tool_map,
                    messages=messages,
                )

        # Max iterations reached
        if self._iteration_task:
            self._iteration_task.status = cl.TaskStatus.FAILED
            await self._task_list.send() if self._task_list else None

        return AgentToolOutput(
            result="Max iterations reached without completing the task",
            success=False,
            iterations_used=validated.max_iterations,
            messages=t.cast(list[dict[str, t.Any]], messages),
        )

    async def _instrumented_plan(
        self,
        messages: list[t.Any],
        iteration: int,
    ) -> StrategyOutput:
        """Run planning with reasoning step instrumentation."""
        strategy_output = await self.strategy.plan(
            messages=messages,
            tools=self.tools,
            parallel_tool_calls=self.parallel_tool_calls,
        )

        # Show reasoning step if there's reasoning content
        if self.show_reasoning and strategy_output.messages:
            for msg in strategy_output.messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    async with cl.Step(name="Reasoning", type="llm") as step:
                        step.output = msg.get("content", "")

        return strategy_output

    async def _instrumented_tool_execution(
        self,
        tool_calls: list[ToolCall],
        tool_map: dict[str, BaseTool[t.Any, t.Any]],
        messages: list[t.Any],
    ) -> None:
        """Execute tool calls with step instrumentation."""
        import json

        # Add assistant message with tool calls
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

        async def execute_with_step(tool_call: ToolCall) -> tuple[str, str]:
            """Execute a single tool with Chainlit step."""
            tool_name = tool_call.tool_name
            tool_input = tool_call.parsed

            tool = tool_map.get(tool_name.upper())

            if self.show_tool_calls:
                async with cl.Step(name=tool_name, type="tool") as step:
                    step.input = convert_for_chainlit(safe_copy(tool_input))

                    if tool is None:
                        result = json.dumps({"error": f"Unknown tool '{tool_name}'"})
                        step.is_error = True
                        step.output = result
                    else:
                        try:
                            output = await tool.acall(tool_input)
                            result = output.model_dump_json()
                            step.output = convert_for_chainlit(safe_copy(output))
                        except Exception as e:
                            result = json.dumps({"error": f"Error: {e}"})
                            step.is_error = True
                            step.output = result
                            logger.error(f"Tool execution error: {e}")

                    return (tool_call.id, result)
            else:
                # Execute without step UI
                if tool is None:
                    return (
                        tool_call.id,
                        json.dumps({"error": f"Unknown tool '{tool_name}'"}),
                    )
                try:
                    output = await tool.acall(tool_input)
                    return (tool_call.id, output.model_dump_json())
                except Exception as e:
                    return (tool_call.id, json.dumps({"error": f"Error: {e}"}))

        # Execute tools (sequentially to maintain step order in UI)
        results = []
        for tc in tool_calls:
            result = await execute_with_step(tc)
            results.append(result)

        # Add tool results to messages
        for tool_call_id, tool_result in results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result,
                }
            )


def create_instrumented_agent(
    agent: AgentTool,
    show_reasoning: bool = True,
    show_tool_calls: bool = True,
    show_task_list: bool = True,
) -> ChainlitInstrumentedAgent:
    """
    Wrap an AgentTool with Chainlit instrumentation.

    Args:
        agent: The AgentTool to wrap
        show_reasoning: Show reasoning/thinking steps
        show_tool_calls: Show each tool call as a step
        show_task_list: Show Chainlit task list

    Returns:
        Instrumented agent with Chainlit UI integration
    """
    return ChainlitInstrumentedAgent(
        agent=agent,
        show_reasoning=show_reasoning,
        show_tool_calls=show_tool_calls,
        show_task_list=show_task_list,
    )

"""
Direct Strategy - Single LLM call with tool_calling mode.

The simplest planning strategy: pass messages to LLM with tools,
let the LLM directly select and parameterize tools in one step.

Best for:
- Simple tasks with clear tool mappings
- When reasoning overhead isn't needed
- Fast iteration cycles
"""

import typing as t

from openai.types.chat import ChatCompletionMessageParam

from agents.agent_tool.base_strategy import PlanningStrategy, StrategyOutput
from agents.configs import get_agent_tool_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool

# Load templates
_templates = get_agent_tool_template_module("direct_strategy.jinja")


class DirectStrategy(PlanningStrategy):
    """
    Direct strategy: Single LLM call with tool_calling mode.

    Flow:
        messages → LLM (tool_calling) → tool_calls → execute

    Best for:
        - Simple tasks with clear tool mappings
        - When reasoning overhead isn't needed
        - Fast iteration cycles

    The LLM directly selects and parameterizes tools in one step.

    Return behavior:
        - No tool_calls → finished=True, success=False (LLM didn't call any tool)
        - FINISH tool called → finished=True, success=<from FINISH args>
        - Other tools called → finished=False, tool_calls=[...] (continue loop)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str | None = None,
        direct_prompt: str | None = None,
        finish_tool_name: str = "finish",
    ):
        """
        Initialize DirectStrategy.

        Args:
            llm_client: LLM client for generation
            model: Optional model override
            direct_prompt: Custom prompt for tool selection (uses template if None)
            finish_tool_name: Name of the tool that signals task completion
        """
        self.llm_client = llm_client
        self.model = model
        self.direct_prompt = direct_prompt
        self.finish_tool_name = finish_tool_name

    def _get_direct_prompt(self, tools: list[BaseTool[t.Any, t.Any]]) -> str:
        """Get the action prompt from template or custom."""
        if self.direct_prompt:
            return self.direct_prompt
        tool_names = [tool.name for tool in tools]
        return _templates.direct_prompt(tool_names=tool_names)

    async def plan(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool = True,
    ) -> StrategyOutput:
        """
        Generate next actions via single LLM call.

        Args:
            messages: Current conversation history
            tools: Available tools (including finish tool)
            parallel_tool_calls: Allow LLM to return multiple tool calls

        Returns:
            StrategyOutput with tool_calls or finished status
        """
        messages = list(messages) + [
            {"role": "user", "content": self._get_direct_prompt(tools)}
        ]
        response = await self.llm_client.agenerate(
            messages=messages,
            model=self.model,
            mode="tool_calling",
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        # No tool calls → treat as finished with no success
        # LLM chose not to call any tools including finish tool
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

        # Return tool calls for execution (pass ToolCall objects directly)
        return StrategyOutput(tool_calls=response.tool_calls)

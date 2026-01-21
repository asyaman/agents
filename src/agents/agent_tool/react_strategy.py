"""
React Strategy - Reason-Act-Observe pattern.

The ReAct pattern separates reasoning from action:
1. Reasoning phase: Analyze state, plan steps, identify IMMEDIATE next action
2. Action phase: Select tool(s) for ONLY the next step
3. (AgentTool executes and adds results to messages)
4. Next iteration: Re-reason with new information

Best for:
- Complex multi-step tasks
- Tasks requiring adaptation based on intermediate results
- When explicit reasoning improves accuracy

Reference: "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
"""

import typing as t

from openai.types.chat import ChatCompletionMessageParam

from agents.agent_tool.base_strategy import PlanningStrategy, StrategyOutput
from agents.configs import get_agent_tool_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool

# Load templates
_templates = get_agent_tool_template_module("react_strategy.jinja")


class ReactStrategy(PlanningStrategy):
    """
    React strategy: Reason-Act-Observe pattern.

    Flow per iteration:
        1. Reasoning: Analyze state, plan steps, identify IMMEDIATE next action
        2. Action: Select tool(s) for ONLY the next step
        3. (AgentTool executes and adds results to messages)
        4. Next iteration: Re-reason with new information

    Best for:
        - Complex multi-step tasks
        - Tasks requiring adaptation based on intermediate results
        - When explicit reasoning improves accuracy

    Key insight: The action phase focuses on the IMMEDIATE next step,
    not the entire plan. This allows the agent to adapt as it learns
    from each step's results.

    Return behavior (after action phase):
        - No tool_calls → finished=True, success=False (LLM didn't call any tool)
        - FINISH tool called → finished=True, success=<from FINISH args>
        - Other tools called → finished=False, tool_calls=[...] (continue loop)

    Note: Reasoning is always included in output messages, even on finish.

    Example:
        Iteration 1:
            Reasoning: "Need to: 1) Get users, 2) Update profiles, 3) Send emails.
                       First, I need the user list."
            Action: get_users(filter="active")

        Iteration 2:
            Reasoning: "Got 5 users. Now I should update their profiles."
            Action: update_profile(user_id=1, data=...)

        Iteration 3:
            Reasoning: "Profiles updated. Now send confirmation emails."
            Action: send_email(to=..., body=...)
    """

    def __init__(
        self,
        action_client: LLMClient,
        action_model: str | None = None,
        reasoning_client: LLMClient | None = None,
        reasoning_model: str | None = None,
        reasoning_prompt: str | None = None,
        action_prompt: str | None = None,
        finish_tool_name: str = "finish",
    ):
        """
        Initialize ReactStrategy.

        Args:
            action_client: LLM client for action phase (tool selection)
            action_model: Model for action phase (uses client default if None)
            reasoning_client: LLM client for reasoning phase (uses action_client if None)
            reasoning_model: Model for reasoning phase (uses action_model if None)
            reasoning_prompt: Custom prompt for reasoning phase (uses template if None)
            action_prompt: Custom prompt for action phase (uses template if None)
            finish_tool_name: Name of the tool that signals task completion
        """
        self.action_client = action_client
        self.action_model = action_model
        # Default reasoning to use same client/model as action if not specified
        self.reasoning_client = reasoning_client or action_client
        self.reasoning_model = reasoning_model or action_model
        self.reasoning_prompt = reasoning_prompt
        self.action_prompt = action_prompt
        self.finish_tool_name = finish_tool_name

    def _get_reasoning_prompt(self, tools: list[BaseTool[t.Any, t.Any]]) -> str:
        """Get the reasoning prompt from template or custom."""
        if self.reasoning_prompt:
            return self.reasoning_prompt
        tool_names = [tool.name for tool in tools]
        return _templates.reasoning_prompt(tool_names=tool_names)

    def _get_action_prompt(self) -> str:
        """Get the action prompt from template or custom."""
        if self.action_prompt:
            return self.action_prompt
        return _templates.action_prompt()

    async def plan(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool[t.Any, t.Any]],
        parallel_tool_calls: bool = True,
    ) -> StrategyOutput:
        """
        Generate next actions using Reason-Act pattern.

        Args:
            messages: Current conversation history
            tools: Available tools (including finish tool)
            parallel_tool_calls: Allow LLM to return multiple tool calls

        Returns:
            StrategyOutput with reasoning messages, tool_calls, and/or finished status
        """
        # Phase 1: Reasoning (text response)
        # Ask LLM to think about current state and what the next step should be
        reasoning_messages = list(messages) + [
            {"role": "user", "content": self._get_reasoning_prompt(tools)}
        ]

        reasoning_response = await self.reasoning_client.agenerate(
            messages=reasoning_messages,
            model=self.reasoning_model,
            mode="text",
        )
        reasoning = reasoning_response.content or ""

        # Phase 2: Action selection (tool_calling)
        # Based on reasoning, select tool(s) for the IMMEDIATE next step only
        action_messages = list(messages) + [
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": self._get_action_prompt()},
        ]

        response = await self.action_client.agenerate(
            messages=action_messages,
            model=self.action_model,
            mode="tool_calling",
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Include reasoning in output messages (becomes part of conversation)
        output_messages: list[ChatCompletionMessageParam] = [
            {"role": "assistant", "content": reasoning}
        ]

        # No tool calls → treat as finished with no success
        # LLM chose not to call any tools including finish tool
        if not response.tool_calls:
            return StrategyOutput(
                messages=output_messages,
                finished=True,
                success=False,
                result=response.finish_reason
                or reasoning + " No tool calls returned by LLM",
            )

        # Check for finish tool
        for tc in response.tool_calls:
            if tc.tool_name.upper() == self.finish_tool_name.upper():
                return StrategyOutput(
                    messages=output_messages,
                    finished=True,
                    success=tc.arguments.get("success", True),
                    result=tc.arguments.get("result", "Task completed"),
                )

        # Return tool calls for immediate next step (pass ToolCall objects directly)
        return StrategyOutput(
            messages=output_messages,
            tool_calls=response.tool_calls,
        )

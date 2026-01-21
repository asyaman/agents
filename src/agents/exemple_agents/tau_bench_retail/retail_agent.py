"""
Retail Customer Service Agent using AgentTool with TauBench tools.

This module provides a pre-configured agent for handling retail customer service
tasks with appropriate escalation rules.
"""

from agents.agent_tool.agent_tool import AgentTool, AgentToolInput, AgentToolOutput
from agents.agent_tool.base_strategy import PlanningStrategy
from agents.exemple_agents.tau_bench_retail.tools import (
    Calculate,
    CancelPendingOrder,
    ExchangeDeliveredOrderItems,
    FindOrderIdsByUserId,
    FindUserIdByEmail,
    FindUserIdByNameZip,
    GetOrderDetails,
    GetProductDetails,
    GetUserDetails,
    ListAllProductTypes,
    ModifyPendingOrderAddress,
    ModifyPendingOrderItems,
    ModifyPendingOrderPayment,
    ModifyUserAddress,
    ReturnDeliveredOrderItems,
    TransferToHumanAgents,
)
from agents.exemple_agents.tau_bench_retail.agent_tools.agents_tau_bench_tool import (
    create_tau_bench_tool,
)
from agents.tools_core.base_tool import BaseTool


# All available TauBench retail tools
RETAIL_TOOL_CLASSES = [
    Calculate,
    CancelPendingOrder,
    ExchangeDeliveredOrderItems,
    FindOrderIdsByUserId,
    FindUserIdByEmail,
    FindUserIdByNameZip,
    GetOrderDetails,
    GetProductDetails,
    GetUserDetails,
    ListAllProductTypes,
    ModifyPendingOrderAddress,
    ModifyPendingOrderItems,
    ModifyPendingOrderPayment,
    ModifyUserAddress,
    ReturnDeliveredOrderItems,
    TransferToHumanAgents,
]


# Guidance messages for retail customer service
RETAIL_GUIDANCE_MESSAGES = [
    """CRITICAL ESCALATION RULES:
- If the customer is transferred to a human agent, HALT immediately.
  Do not proceed with solving the objective. Call `finish` with success=True
  and indicate the transfer was completed.""",
    """AMBIGUITY HANDLING - Escalate to human agent when:
- The customer requests cancellation or return, but MULTIPLE orders contain the same product
- The customer requests a PARTIAL return (not the entire order)
- The customer's intent is unclear or could be interpreted multiple ways
- Required information is missing and cannot be inferred

Do NOT attempt to guess or make assumptions. Transfer to human for clarification.""",
]


def create_retail_tools() -> list[BaseTool]:
    """Create all retail customer service tools."""
    return [create_tau_bench_tool(tool) for tool in RETAIL_TOOL_CLASSES]


def create_retail_agent(
    strategy: PlanningStrategy,
    guidance_messages: list[str] | None = None,
    parallel_tool_calls: bool = True,
    include_finish_tool: bool = True,
) -> AgentTool:
    """
    Create a retail customer service agent.

    Args:
        strategy: Planning strategy for the agent (e.g., DirectStrategy, ReactStrategy)
        guidance_messages: Custom guidance messages (defaults to RETAIL_GUIDANCE_MESSAGES)
        parallel_tool_calls: Allow parallel tool execution (default True)
        include_finish_tool: Include the finish tool (default True)

    Returns:
        Configured AgentTool for retail customer service
    """
    tools = create_retail_tools()

    agent = AgentTool(
        tools=tools,
        strategy=strategy,
        guidance_messages=guidance_messages or RETAIL_GUIDANCE_MESSAGES,
        parallel_tool_calls=parallel_tool_calls,
        include_finish_tool=include_finish_tool,
    )
    return agent


__all__ = [
    "create_retail_agent",
    "create_retail_tools",
    "RETAIL_TOOL_CLASSES",
    "RETAIL_GUIDANCE_MESSAGES",
    "AgentToolInput",
    "AgentToolOutput",
]

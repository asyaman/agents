"""
Retail Customer Service Agent configuration for Chainlit.

This module provides the AgentConfig for the retail customer service agent,
which handles order inquiries, returns, exchanges, and account management.

LLM Client Configuration:
    The agent supports configurable LLM clients for different purposes:
    - tool_client: Client for tool operations (order lookups, modifications)
    - action_client: Client for deciding which tools to call (ReactStrategy action)
    - reasoning_client: Client for reasoning/thinking steps (ReactStrategy reasoning)

    If not provided, falls back to the tool_client.
"""

from agents.agent_tool.agent_tool import AgentTool
from agents.agent_tool.react_strategy import ReactStrategy
from agents.exemple_agents.tau_bench_retail.retail_agent import (
    RETAIL_GUIDANCE_MESSAGES,
    create_retail_tools,
)
from agents.llm_core.llm_client import LLMClient, create_azure_client
from agents.webapp.runner.agent_config import AgentConfig


def create_retail_webapp_agent(
    tool_client: LLMClient,
    action_client: LLMClient | None = None,
    reasoning_client: LLMClient | None = None,
    escalation_threshold: int = 3,
    **kwargs,
) -> AgentTool:
    """
    Factory function to create a retail customer service agent.

    Args:
        tool_client: LLM client for tool operations (order lookups, modifications)
        action_client: Client for action decisions (defaults to tool_client)
        reasoning_client: Client for reasoning steps (defaults to tool_client)
        escalation_threshold: Number of failed attempts before suggesting escalation
        **kwargs: Additional settings (ignored)

    Returns:
        Configured AgentTool for retail customer service
    """
    # Use provided clients or fall back to tool_client
    effective_action_client = action_client or tool_client
    effective_reasoning_client = reasoning_client or tool_client

    # Create retail tools (no LLM client needed for these tools)
    tools = create_retail_tools()

    # Build guidance messages
    guidance_messages = list(RETAIL_GUIDANCE_MESSAGES)

    # Add escalation threshold guidance
    guidance_messages.append(
        f"Escalation threshold: {escalation_threshold}. After {escalation_threshold} "
        "failed attempts to resolve an issue, suggest transferring to a human agent."
    )

    # Create strategy with separate action and reasoning clients
    strategy = ReactStrategy(
        action_client=effective_action_client,
        reasoning_client=effective_reasoning_client,
    )

    # Create agent
    return AgentTool(
        tools=tools,
        strategy=strategy,
        guidance_messages=guidance_messages,
        parallel_tool_calls=True,  # Allow parallel lookups for efficiency
    )


# Welcome message with instructions
WELCOME_MESSAGE = """
## Welcome to Retail Customer Service

I'm here to help you with your orders and account.

### What I can help with:
- **Order Status** - Check the status of your orders
- **Returns** - Process returns for delivered items
- **Exchanges** - Exchange items for different products
- **Order Modifications** - Update pending orders (items, address, payment)
- **Account Updates** - Update your address information

### To get started:
Tell me what you need help with, for example:
> *"I'd like to return an item from my recent order"*
> *"Can you check the status of my order?"*
> *"I need to change the shipping address on my pending order"*

### Important:
- Please have your email or order information ready
- For complex issues, I may transfer you to a human agent

---
**How can I assist you today?**
""".strip()


# Chainlit settings widgets
SETTINGS_WIDGETS = [
    {
        "type": "slider",
        "id": "escalation_threshold",
        "label": "Escalation Threshold",
        "initial": 3,
        "min": 1,
        "max": 5,
        "step": 1,
        "description": "Number of failed attempts before suggesting human transfer",
    },
]


# Agent configuration
RETAIL_AGENT_CONFIG = AgentConfig(
    name="Retail Customer Service Agent",
    description="AI-powered customer service for orders, returns, exchanges, and account management",
    welcome_message=WELCOME_MESSAGE,
    agent_factory=create_retail_webapp_agent,
    tool_wrappers={},  # No special wrappers needed for retail tools
    settings_widgets=SETTINGS_WIDGETS,
    default_settings={
        "escalation_threshold": 3,
    },
    # LLM client configuration for different roles
    llm_clients={
        "tool": create_azure_client(default_model="gpt-5-mini", async_client=True),
        "action": create_azure_client(default_model="gpt-5-mini", async_client=True),
        "reasoning": create_azure_client(default_model="gpt-5-mini", async_client=True),
    },
    max_iterations=25,  # Retail tasks may require more iterations
    show_reasoning=True,
    show_task_list=True,
)

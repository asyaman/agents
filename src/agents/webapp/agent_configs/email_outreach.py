"""
Email Outreach Agent configuration for Chainlit.

This module provides the AgentConfig for the email outreach agent,
which can draft and send personalized sales emails.

LLM Client Configuration:
    The agent supports configurable LLM clients for different purposes:
    - tool_client: Client for tool operations (email drafting, research)
    - action_client: Client for deciding which tools to call (ReactStrategy action)
    - reasoning_client: Client for reasoning/thinking steps (ReactStrategy reasoning)

    If not provided, falls back to the main llm_client.
"""

from agents.agent_tool.agent_tool import AgentTool
from agents.agent_tool.react_strategy import ReactStrategy
from agents.llm_core.llm_client import LLMClient, create_azure_client
from agents.exemple_agents.email_outreach.email_outreach_agent import (
    EMAIL_OUTREACH_GUIDANCE_MESSAGES,
    create_email_outreach_tools,
)
from agents.webapp.runner.agent_config import AgentConfig
from agents.webapp.tool_wrappers.human_approval import ChainlitHumanMailContentApproval


def create_email_outreach_agent(
    tool_client: LLMClient,
    action_client: LLMClient | None = None,
    reasoning_client: LLMClient | None = None,
    guidance: str = "",
    tone: str = "professional",
    max_retries: int = 3,
    **kwargs,
) -> AgentTool:
    """
    Factory function to create an email outreach agent.

    Args:
        tool_client: LLM client for tool operations (email drafting, research)
        action_client: Client for action decisions (defaults to tool_client)
        reasoning_client: Client for reasoning steps (defaults to tool_client)
        guidance: Additional guidance/instructions for the agent
        tone: Email tone (professional, friendly, casual)
        max_retries: Maximum retry attempts for email drafts
        **kwargs: Additional settings (ignored)

    Returns:
        Configured AgentTool for email outreach
    """
    # Use provided clients or fall back to tool_client
    effective_tool_client = tool_client
    effective_action_client = action_client or tool_client
    effective_reasoning_client = reasoning_client or tool_client

    # Create tools with tool-specific client
    tools = create_email_outreach_tools(effective_tool_client)

    # Build guidance messages
    guidance_messages = list(EMAIL_OUTREACH_GUIDANCE_MESSAGES)

    # Add tone instruction
    tone_instructions = {
        "professional": "Use a professional, business-appropriate tone. Be formal but not stiff.",
        "friendly": "Use a warm, friendly tone. Be personable while maintaining professionalism.",
        "casual": "Use a casual, conversational tone. Be approachable and relaxed.",
    }
    guidance_messages.append(
        tone_instructions.get(tone, tone_instructions["professional"])
    )

    # Add custom guidance if provided
    if guidance and guidance.strip():
        guidance_messages.append(f"Additional instructions: {guidance}")

    # Add retry limit
    guidance_messages.append(
        f"Maximum revision attempts: {max_retries}. After {max_retries} retries, "
        "suggest stopping to avoid excessive iterations."
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
        parallel_tool_calls=False,  # Sequential for email workflow
    )


# Welcome message with instructions
WELCOME_MESSAGE = """
## Welcome to the Email Outreach Agent

I'll help you draft personalized sales emails for potential customers.

### How it works:
1. **Provide contact info** - Give me a name and email address
2. **I'll research** - I'll look up the company from the email domain
3. **Draft email** - I'll write a personalized outreach email
4. **Your approval** - Review and approve, request changes, or cancel

### To get started:
Just tell me who you'd like to reach out to, for example:
> *"I want to reach out to Jane Smith at jane@acme.com"*

### Settings
Click the settings icon in the chat to customize:
- **Guidance**: Add specific instructions
- **Tone**: Professional, friendly, or casual
- **Max Retries**: How many revision attempts

---
**Ready when you are!**
""".strip()


# Chainlit settings widgets
SETTINGS_WIDGETS = [
    {
        "type": "text",
        "id": "guidance",
        "label": "Additional Guidance",
        "initial": "",
        "description": "Extra instructions for the agent (e.g., 'Focus on AI automation benefits')",
        "multiline": True,
    },
    {
        "type": "select",
        "id": "tone",
        "label": "Email Tone",
        "values": ["professional", "friendly", "casual"],
        "initial": "professional",
        "description": "The tone/style for drafted emails",
    },
    {
        "type": "slider",
        "id": "max_retries",
        "label": "Max Revisions",
        "initial": 3,
        "min": 1,
        "max": 5,
        "step": 1,
        "description": "Maximum number of email revision attempts",
    },
]


# Agent configuration
EMAIL_OUTREACH_CONFIG = AgentConfig(
    name="Email Outreach Agent",
    description="AI-powered sales email drafting with company research and human approval workflow",
    welcome_message=WELCOME_MESSAGE,
    agent_factory=create_email_outreach_agent,
    tool_wrappers={
        "human_approval": ChainlitHumanMailContentApproval,
    },
    settings_widgets=SETTINGS_WIDGETS,
    default_settings={
        "guidance": "",
        "tone": "professional",
        "max_retries": 3,
    },
    # LLM client configuration for different roles
    # Each role can use a different model/provider
    llm_clients={
        "tool": create_azure_client(default_model="gpt-5-mini", async_client=True),
        "action": create_azure_client(default_model="gpt-5-mini", async_client=True),
        "reasoning": create_azure_client(default_model="gpt-5-mini", async_client=True),
    },
    max_iterations=20,
    show_reasoning=True,
    show_task_list=True,
)

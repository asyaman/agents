"""
Agent configuration schema for Chainlit.

Defines how an agent should be configured for the generic Chainlit runner.
Each agent provides an AgentConfig that specifies:
- Display information (name, description, welcome message)
- Factory function to create the AgentTool
- LLM client configuration for different roles (tool, action, reasoning)
- Tool wrappers for Chainlit-aware tools
- Settings schema for user customization
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agents.agent_tool.agent_tool import AgentTool
from agents.llm_core.llm_client import LLMClient


# Type alias for LLM client or factory function
LLMClientOrFactory = LLMClient | Callable[[], LLMClient]


class AgentFactory(Protocol):
    """Protocol for agent factory functions."""

    def __call__(
        self,
        **settings: Any,
    ) -> AgentTool:
        """
        Create an AgentTool instance.

        Args:
            **settings: User-configured settings from Chainlit ChatSettings
                Includes tool_client, action_client, reasoning_client
                from AgentConfig.llm_clients

        Returns:
            Configured AgentTool instance
        """
        ...


@dataclass
class AgentConfig:
    """
    Configuration for running an agent in Chainlit.

    This separates agent logic (AgentTool) from UI configuration,
    allowing the same AgentRunner to work with any agent.

    Example:
        config = AgentConfig(
            name="Email Outreach",
            description="AI-powered sales emails",
            welcome_message="Hello! I'll help you write emails...",
            agent_factory=create_email_agent,
            tool_wrappers={"human_approval": ChainlitHumanMailContentApproval},
            settings_widgets=[
                {"type": "text", "id": "guidance", "label": "Guidance"},
            ],
        )
    """

    # Display information
    name: str
    description: str
    welcome_message: str

    # Factory function to create the agent
    # Signature: (llm_client: LLMClient, **settings) -> AgentTool
    agent_factory: AgentFactory

    # Tools that need Chainlit UI wrappers
    # Maps tool_name (lowercase) -> wrapper_class
    # Wrapper class should accept (original_tool) in __init__
    tool_wrappers: dict[str, type] = field(default_factory=dict)

    # Settings widgets for Chainlit ChatSettings
    # Each dict should have: id, type, label, and type-specific options
    # Supported types: text, select, slider, switch, tags
    # Example: {"type": "text", "id": "guidance", "label": "Guidance", "initial": ""}
    settings_widgets: list[dict[str, Any]] = field(default_factory=list)

    # Default settings values (used when no ChatSettings configured)
    default_settings: dict[str, Any] = field(default_factory=dict)

    # LLM client configuration for different roles
    # Maps role name -> LLMClient instance or callable that returns LLMClient
    # Supported roles: "default", "tool", "action", "reasoning"
    # Example:
    #   llm_clients={
    #       "default": create_azure_client(default_model="gpt-4o", async_client=True),
    #       "tool": create_azure_client(default_model="gpt-4o", async_client=True),
    #       "action": create_azure_client(default_model="gpt-4o-mini", async_client=True),
    #       "reasoning": create_azure_client(default_model="gpt-4o", async_client=True),
    #   }
    llm_clients: dict[str, LLMClientOrFactory] = field(default_factory=dict)

    # Optional: Maximum iterations for agent
    max_iterations: int = 15

    # Optional: Show reasoning steps in UI (for ReactStrategy)
    show_reasoning: bool = True

    # Optional: Show task list in UI
    show_task_list: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Agent name is required")
        if not self.agent_factory:
            raise ValueError("Agent factory is required")

    def get_llm_client(self, role: str = "default") -> LLMClient | None:
        """
        Get the LLM client for a specific role.

        Args:
            role: The role name ("default", "tool", "action", "reasoning")

        Returns:
            LLMClient instance or None if not configured
        """
        client_or_factory = self.llm_clients.get(role)
        if client_or_factory is None:
            return None
        if callable(client_or_factory) and not isinstance(client_or_factory, LLMClient):
            # It's a factory function, call it to get the client
            return client_or_factory()
        return client_or_factory

"""
Agent registry for Chainlit webapp.

Central registry of all available agent configurations.
Add new agents here to make them available in the UI.
"""

from agents.webapp.agent_configs.email_outreach import EMAIL_OUTREACH_CONFIG
from agents.webapp.agent_configs.retail_agent import RETAIL_AGENT_CONFIG
from agents.webapp.runner.agent_config import AgentConfig

# Registry of all available agents
# Key: agent name (used in AGENT_NAME env var)
# Value: AgentConfig instance
AGENT_REGISTRY: dict[str, AgentConfig] = {
    "email_outreach": EMAIL_OUTREACH_CONFIG,
    "retail": RETAIL_AGENT_CONFIG,
}


def get_agent_config(name: str) -> AgentConfig:
    """
    Get agent configuration by name.
    """
    name_lower = name.lower()
    if name_lower not in AGENT_REGISTRY:
        available = ", ".join(sorted(AGENT_REGISTRY.keys()))
        raise ValueError(f"Unknown agent: '{name}'. Available agents: {available}")
    return AGENT_REGISTRY[name_lower]


def list_agents() -> list[str]:
    """
    List all available agent names.
    """
    return sorted(AGENT_REGISTRY.keys())


def get_agent_info() -> list[dict[str, str]]:
    """
    Get info (name, description) about all available agents.

    """
    return [
        {
            "name": name,
            "description": config.description,
        }
        for name, config in sorted(AGENT_REGISTRY.items())
    ]

"""
Webapp - Generic Chainlit-based UI for AgentTool.

This module provides a standardized way to run any AgentTool with a Chainlit UI.
The architecture separates:
- Agent logic (AgentTool, strategies, tools) - reusable across UIs
- Agent configuration (AgentConfig) - per-agent customization
- UI runner (AgentRunner) - generic Chainlit integration

Usage:
    # Run with default agent (set via AGENT_NAME env var)
    chainlit run app.py

    # Or set agent programmatically
    AGENT_NAME=email_outreach chainlit run app.py
"""

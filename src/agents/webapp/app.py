"""
Chainlit application entry point.

This is the main entry point for the Chainlit webapp.
It uses the generic AgentRunner to run any configured agent.

Usage:
    # Run with default agent (email_outreach)
    cd src/agents/webapp
    chainlit run app.py --port 9001

    # Run with a different agent
    AGENT_NAME=research chainlit run app.py --port 9001

Environment Variables:
    AGENT_NAME: Name of the agent to run (default: email_outreach)
    DATABASE_URL: SQLAlchemy URL for chat history (default: sqlite)
    OPENAI_API_KEY: Required for LLM calls
    TAVILY_API_KEY: Optional, for web search functionality
"""

import chainlit as cl
from loguru import logger

from agents.webapp.config import settings, ensure_data_dir
from agents.webapp.agent_configs.registry import get_agent_config, list_agents
from agents.webapp.runner.agent_runner import AgentRunner

# Ensure data directory exists for SQLite
ensure_data_dir()

# Get agent configuration
try:
    agent_config = get_agent_config(settings.AGENT_NAME)
    logger.info(f"Loaded agent: {agent_config.name}")
except ValueError as e:
    logger.error(str(e))
    logger.info(f"Available agents: {list_agents()}")
    raise

# Create runner
runner = AgentRunner(agent_config)


# Optional: SQLite data layer for chat history persistence
# Uncomment to enable chat history
# from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
# @cl.data_layer
# def get_data_layer():
#     return SQLAlchemyDataLayer(conninfo=settings.DATABASE_URL)


@cl.on_chat_start
async def on_chat_start():
    """Handle new chat session."""
    await runner.on_chat_start()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming message."""
    await runner.on_message(message)


@cl.on_settings_update
async def on_settings_update(new_settings: dict):
    """Handle settings changes."""
    await runner.on_settings_update(new_settings)


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Handle chat resume (if data layer enabled)."""
    logger.info(f"Resuming chat thread: {thread.get('id', 'unknown')}")
    # Recreate agent state
    await runner.on_chat_start()


# Development: Log startup info
if settings.is_development:
    logger.info("=" * 50)
    logger.info("Running in DEVELOPMENT mode")
    logger.info(f"Agent: {agent_config.name}")
    logger.info(f"Database: {settings.DATABASE_URL}")
    logger.info("=" * 50)

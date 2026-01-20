"""
Generic Chainlit agent runner.

This module provides a generic runner that works with ANY AgentTool.
The runner handles:
- Agent initialization with user settings
- Chat message handling
- UI feedback (steps, task list)
- Tool wrapper application

The runner is configured via AgentConfig, which allows per-agent
customization without changing the runner code.
"""

import typing as t

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch, Tags, TextInput
from loguru import logger

from agents.agent_tool.agent_tool import AgentTool, AgentToolInput
from agents.webapp.runner.agent_config import AgentConfig
from agents.webapp.runner.chainlit_instrumentation import create_instrumented_agent


def _create_widget(widget_config: dict[str, t.Any]) -> t.Any:
    """Create a Chainlit input widget from configuration dict."""
    widget_type = widget_config.get("type", "text")
    widget_id = widget_config["id"]
    label = widget_config.get("label", widget_id)

    if widget_type == "text":
        return TextInput(
            id=widget_id,
            label=label,
            initial=widget_config.get("initial", ""),
            description=widget_config.get("description"),
            multiline=widget_config.get("multiline", False),
        )
    elif widget_type == "select":
        return Select(
            id=widget_id,
            label=label,
            values=widget_config.get("values", []),
            initial=widget_config.get("initial"),
            description=widget_config.get("description"),
        )
    elif widget_type == "slider":
        return Slider(
            id=widget_id,
            label=label,
            initial=widget_config.get("initial", 0),
            min=widget_config.get("min", 0),
            max=widget_config.get("max", 100),
            step=widget_config.get("step", 1),
            description=widget_config.get("description"),
        )
    elif widget_type == "switch":
        return Switch(
            id=widget_id,
            label=label,
            initial=widget_config.get("initial", False),
            description=widget_config.get("description"),
        )
    elif widget_type == "tags":
        return Tags(
            id=widget_id,
            label=label,
            initial=widget_config.get("initial", []),
            description=widget_config.get("description"),
        )
    else:
        raise ValueError(f"Unknown widget type: {widget_type}")


class AgentRunner:
    """
    Generic runner that works with ANY AgentTool.

    The runner provides a standardized Chainlit experience:
    - Welcome message on chat start
    - Optional settings panel for user customization
    - Agent execution with UI feedback (steps, task list)
    - Tool wrapper application for interactive tools

    Usage:
        config = AgentConfig(...)
        runner = AgentRunner(config)

        @cl.on_chat_start
        async def on_chat_start():
            await runner.on_chat_start()

        @cl.on_message
        async def on_message(message):
            await runner.on_message(message)
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the runner with an agent configuration.

        Args:
            config: AgentConfig defining the agent behavior and UI
        """
        self.config = config

    async def on_chat_start(self) -> None:
        """
        Handle chat start event.

        Creates the agent, applies settings, and sends welcome message.
        """
        logger.info(f"Starting chat with agent: {self.config.name}")

        # Show settings panel if widgets defined
        settings_dict: dict[str, t.Any] = dict(self.config.default_settings)

        if self.config.settings_widgets:
            widgets = [_create_widget(w) for w in self.config.settings_widgets]
            settings = await cl.ChatSettings(widgets).send()
            if settings:
                settings_dict.update(settings)

        cl.user_session.set("settings", settings_dict)

        # Create agent using factory
        agent = await self._create_agent(settings_dict)
        cl.user_session.set("agent", agent)

        # Send welcome message
        await cl.Message(content=self.config.welcome_message).send()

    async def on_message(self, message: cl.Message) -> None:
        """
        Handle incoming chat message.

        Runs the agent and displays results with UI feedback.
        The ChainlitInstrumentedAgent handles:
        - Reasoning steps display
        - Tool call steps with input/output
        - Task list progress
        """
        agent: AgentTool | None = cl.user_session.get("agent")
        if not agent:
            await cl.Message(
                content="Error: Agent not initialized. Please refresh the page."
            ).send()
            return

        settings = cl.user_session.get("settings", {})

        logger.info(f"Processing message: {message.content[:50]}...")

        try:
            # Run agent - instrumentation handles step/task visualization
            result = await agent.ainvoke(
                AgentToolInput(
                    objective=message.content,
                    context=settings.get("context"),
                    max_iterations=self.config.max_iterations,
                )
            )

            # Send final result message
            status = "[Success]" if result.success else "[Warning]"
            await cl.Message(
                content=f"{status} {result.result}",
            ).send()

            logger.info(
                f"Agent completed | success={result.success} | iterations={result.iterations_used}"
            )

        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")

            await cl.Message(
                content=f"[Error] An error occurred: {str(e)}",
            ).send()

    async def on_settings_update(self, new_settings: dict[str, t.Any]) -> None:
        """
        Handle settings update event.

        Recreates the agent with new settings.
        """
        logger.info(f"Settings updated: {new_settings}")

        # Merge with defaults
        settings_dict = dict(self.config.default_settings)
        settings_dict.update(new_settings)
        cl.user_session.set("settings", settings_dict)

        # Recreate agent with new settings
        agent = await self._create_agent(settings_dict)
        cl.user_session.set("agent", agent)

        await cl.Message(content="Settings updated.").send()

    async def _create_agent(
        self,
        settings: dict[str, t.Any],
    ) -> AgentTool:
        """
        Create agent instance with tool wrappers and instrumentation applied.

        Args:
            settings: User settings from ChatSettings

        Returns:
            Configured AgentTool with Chainlit tool wrappers and UI instrumentation
        """
        # Build settings with LLM clients from config
        factory_settings = dict(settings)

        # Get LLM clients from config and add to settings
        for role in ("tool", "action", "reasoning"):
            client = self.config.get_llm_client(role)
            if client:
                factory_settings[f"{role}_client"] = client

        # Validate at least one client is configured
        if not any(
            f"{role}_client" in factory_settings
            for role in ("tool", "action", "reasoning")
        ):
            raise ValueError(
                "AgentConfig must define at least one LLM client in llm_clients"
            )

        # Create base agent from factory
        agent = self.config.agent_factory(**factory_settings)

        # Apply Chainlit tool wrappers
        if self.config.tool_wrappers:
            agent = self._apply_tool_wrappers(agent)

        # Wrap with Chainlit UI instrumentation
        instrumented_agent = create_instrumented_agent(
            agent=agent,
            show_reasoning=True,
            show_tool_calls=True,
            show_task_list=self.config.show_task_list,
        )

        return instrumented_agent

    def _apply_tool_wrappers(self, agent: AgentTool) -> AgentTool:
        """
        Replace tools with Chainlit-aware versions.

        Args:
            agent: AgentTool with original tools

        Returns:
            AgentTool with wrapped tools
        """
        for i, tool in enumerate(agent.tools):
            tool_name_lower = tool.name.lower()
            if tool_name_lower in self.config.tool_wrappers:
                wrapper_cls = self.config.tool_wrappers[tool_name_lower]
                logger.debug(f"Wrapping tool '{tool.name}' with {wrapper_cls.__name__}")
                agent.tools[i] = wrapper_cls(tool)

        return agent

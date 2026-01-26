# Chainlit Webapp forhyperions

A generic Chainlit-based web interface for running any AgentTool-based agent.

## Features

- **Generic Agent Runner**: Single Chainlit app that works with any agent
- **Local Storage**: SQLite-based chat history (no external services)
- **No Authentication**: Simplified single-user setup
- **Configurable UI**: Per-agent settings, welcome messages, and customization
- **Tool Wrappers**: Chainlit-aware vershyperionf interactive tools (e.g., human approval with buttons)

## Quick Start

### 1. Install Dependencies

```bash
pip install chainlit loguru pydantic-settings
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. Run the App

```bash
cd src/agents/webapp
AGENT_NAME=email_outreach uv run chainlit run app.py --port 9001
```

Open http://localhost:9001 in your browser.

## Configuration

### Envhyperionnt Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENT_NAME` | Name of agent to run | `email_outreach` |
| `DATABASE_URL` | SQLAlchemy URL for chat history | `sqlite+aiosqlite:///./data/chat_history.db` |
| `ENVIRONMENT` | Environment mode (`local`/`production`) | `local` |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `TAVILY_API_KEY` | Tavily API key for web search | Optional |

### Available Agents

List available agents:
```python
from agents.webapp.agents.registry import list_agents
print(list_agents())  # ['email_outreach']
```

## Architecture

```
webapp/
├── app.py                    # Chainlit entry point
├── config.py                 # Environment settings
├── .chainlit/
│   └── config.toml          # Chainlit UI configuration
├── runner/
│   ├── agent_config.py      # Per-agent configuration schema
│   └── agent_runner.py      # Generic Chainlit runner
├── agents/
│   ├── registry.py          # Agent registry
│   └── email_outreach.py    # Email outreach agent config
├── tool_wrappers/
│   ├── base.py              # Base wrapper class
│   └── human_approval.py    # Chainlit buttons for approval
└── storage/                  # Chat history storage
```

## Adding a New Agent

### 1. Create Agent Configuration

Create a new file in `agents/` (e.g., `agents/research.py`):

```python
from agents.agent_tool.agent_tool import AgentTool
from agents.llm_core.llm_client import LLMClient
from agents.webapp.runner.agent_config import AgentConfig

dhyperionate_research_agent(llm_client: LLMClient, **kwargs) -> AgentTool:
    """Factory function to create the agent."""
    # Create your tools
    tools = [...]

    # Create strategy
    strategy = DirectStrategy(client=llm_client)

    # Return configured agent
    return AgentTool(
        tools=tools,
        strategy=strategy,
        guidance_messages=["You are a research assistant..."],
    )

RESEARCH_CONFIG = AgentConfig(
    name="Research Agent",
    description="AI-powered research assistant",
    welcome_message="Welcome! I can help you research topics.",
    agent_factory=create_research_agent,
    tool_wrappers={},  # Add any Chainlit-aware tool wrappers
    settings_widgets=[
        {
            "type": "text",
            "id": "topic_focus",
            "label": "Topic Focus",
            "initial": "",
        },
    ],
    default_settings={
        "topic_focus": "",
    },
)
```

### 2. Register the Agent

Add to `agents/registry.py`:

```python
from agents.webapp.agents.research import RESEARCH_CONFIG

AGENT_REGISTRY: dict[str, AgentConfig] = {
    "email_outreach": EMAIL_OUTREACH_CONFIG,
    "research": RESEARCH_CONFIG,  # Add your agent
}
```

### 3. Run Your Agent

```bash
AGENT_NAME=research chainlit run app.py --port 9001
```

## Settings Widgets

Configure user-adjustable settings in `settings_widgets`:

```python
settings_widgets = [
    # Text input
    {
        "type": "text",
        "id": "guidance",
        "label": "Additional Guidance",
        "initial": "",
        "multiline": True,
    },
    # Dropdown select
    {
        "type": "select",
        "id": "tone",
        "label": "Tone",
        "values": ["professional", "friendly", "casual"],
        "initial": "professional",
    },
    # Slider
    {
        "type": "slider",
        "id": "max_retries",
        "label": "Max Retries",
        "initial": 3,
        "min": 1,
        "max": 10,
        "step": 1,
    },
    # Toggle switch
    {
        "type": "switch",
        "id": "verbose",
        "label": "Verbose Mode",
        "initial": False,
    },
]
```

## Tool Wrappers

Tool wrappers replace standard tools with Chainlit UI versions:

```python
from agents.webapp.tool_wrappers.base import ChainlitToolWrapper

class ChainlitMyTool(ChainlitToolWrapper):
    """Chainlit-aware version of MyTool."""

    async def ainvoke(self, input: MyToolInput) -> MyToolOutput:
        # Show UI elements using Chainlit
        await cl.Message(content="Processing...").send()

        # Get user input if needed
        response = await cl.AskActionMessage(
            content="Choose an action",
            actions=[
                cl.Action(name="option1", payload={"value": "1"}, label="Option 1"),
                cl.Action(name="option2", payload={"value": "2"}, label="Option 2"),
            ]
        ).send()

        # Return output matching original tool's schema
        return MyToolOutput(result=response["payload"]["value"])
```

Register in your agent config:

```python
EMAIL_OUTREACH_CONFIG = AgentConfig(
    ...
    tool_wrappers={
        "my_tool": ChainlitMyTool,  # Maps tool name to wrapper class
    },
)
```

## Enabling Chat History Persistence

Uncomment the data layer in `app.py`:

```python
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=settings.DATABASE_URL)
```

Install async SQLite driver:

```bash
pip install aiosqlite
```

## UI Customization

Edit `.chainlit/config.toml` to customize:

- Theme colors
- App name and description
- File upload settings
- Feature toggles

See [Chainlit docs](https://docs.chainlit.io/customisation/overview) for full options.

## Development

### Run in Development Mode

```bash
ENVIRONMENT=local chainlit run app.py --port 9001 -w
```

The `-w` flag enables auto-reload on file changes.

### Project Structure

| Component | Purpose |
|-----------|---------|
| `AgentConfig` | Defines agent metadata, factory, settings |
| `AgentRunner` | Generic runner that works with any AgentConfig |
| `Tool Wrappers` | Replace tools with Chainlit UI versions |
| `Registry` | Central index of available agents |

# Agents Package

Core Python package for the agents framework.

## Package Structure

```
agents/
├── llm_core/       # LLM client abstractions
├── tools_core/     # Tool base classes
├── tools/          # Tool implementations
├── agent_tool/     # Agent executor
├── settings.py     # Configuration
├── configs.py      # Template loading
└── utils.py        # Utilities
```

## Module Documentation

- [LLM Core](llm_core/README.md) - Multi-provider LLM client
- [Tools Core](tools_core/README.md) - Tool base classes
- [Tools](tools/README.md) - Available tools
- [Agent Tool](agent_tool/README.md) - Agent executor

## Quick Reference

### Import Patterns

```python
# Agent and strategies
from agents.agent_tool.agent_tool import AgentTool, AgentToolInput
from agents.agent_tool.direct_strategy import DirectStrategy
from agents.agent_tool.react_strategy import ReactStrategy

# LLM clients
from agents.llm_core.llm_client import create_openai_client, create_ollama_client

# Tools
from agents.tools.calculator import Calculator
from agents.tools.tavily import TavilySearch

# Base classes for custom tools
from agents.tools_core.base_tool import BaseTool, create_fn_tool

# Settings
from agents.settings import get_settings, get_api_key
```

### Minimal Example

```python
import asyncio
from agents.agent_tool.agent_tool import AgentTool, AgentToolInput
from agents.agent_tool.direct_strategy import DirectStrategy
from agents.llm_core.llm_client import create_openai_client
from agents.tools.calculator import Calculator

async def main():
    client = create_openai_client()
    agent = AgentTool(
        tools=[Calculator()],
        strategy=DirectStrategy(llm_client=client),
    )
    result = await agent.ainvoke(AgentToolInput(objective="What is 10 * 5?"))
    print(result.result)

asyncio.run(main())
```

## Configuration

Settings are loaded from environment variables or `.env` file:

```python
from agents.settings import get_settings, get_api_key

settings = get_settings()
openai_key = get_api_key("openai")
```

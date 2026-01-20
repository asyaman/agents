# Agents

A Python framework for building autonomous agents with pluggable LLM clients, tools, and planning strategies.

## Features

- **Multi-Provider LLM Support**: 14+ providers (OpenAI, Azure, Ollama, Groq, Together, Mistral, etc.)
- **Typed Tool System**: Pydantic-validated input/output for all tools
- **Pluggable Planning**: DirectStrategy (fast) and ReactStrategy (thoughtful)
- **Recursive Agents**: Sub-agent delegation for complex hierarchical tasks
- **Async-First**: Built for async execution with sync fallbacks

## Project Structure

```
src/agents/
├── llm_core/       # LLM client abstractions
├── tools_core/     # Tool base classes and internal tools
├── tools/          # Concrete tool implementations
├── agent_tool/     # Agent executor and planning strategies
└── settings.py     # Configuration management
```

## Requirements

- Python 3.14+
- See `pyproject.toml` for full dependency list

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd agents

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
# Add other provider keys as needed
```

Supported provider keys:
- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- `GROQ_API_KEY`
- `TOGETHER_API_KEY`
- `MISTRAL_API_KEY`
- `HUGGINGFACE_API_KEY`
- `FIREWORKS_API_KEY`
- `DEEPSEEK_API_KEY`

## Quick Start

```python
import asyncio
from agents.agent_tool.agent_tool import AgentTool, AgentToolInput
from agents.agent_tool.direct_strategy import DirectStrategy
from agents.llm_core.llm_client import create_openai_client
from agents.tools.calculator import Calculator
from agents.tools.tavily import TavilySearch

async def main():
    # Create LLM client
    client = create_openai_client()

    # Create planning strategy
    strategy = DirectStrategy(llm_client=client)

    # Create agent with tools
    agent = AgentTool(
        tools=[Calculator(), TavilySearch()],
        strategy=strategy,
    )

    # Run agent
    result = await agent.ainvoke(
        AgentToolInput(objective="What is 15 * 7? Then search for Python tutorials.")
    )
    print(result.result)

asyncio.run(main())
```

## Development

```bash
# Run tests
pytest

# Run specific test file
pytest src/agents/agent_tool/tests/test_agent_tool.py

# Run with verbose output
pytest -v

# Run only unit tests (skip integration)
pytest -m "not integration"
```

## Documentation

- [LLM Core](src/agents/llm_core/README.md) - LLM client system
- [Tools Core](src/agents/tools_core/README.md) - Tool base classes
- [Agent Tool](src/agents/agent_tool/README.md) - Agent executor
- [Tools](src/agents/tools/README.md) - Available tools

## License

MIT

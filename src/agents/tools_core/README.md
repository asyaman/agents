# Tools Core

Base classes and utilities for building typed tools with Pydantic validation.

## Concept

Tools are the building blocks that agents use to interact with the world. Every tool:
- Has explicit Pydantic input/output models
- Supports both sync and async execution
- Provides a description for LLM consumption

## Components

| File | Purpose |
|------|---------|
| `base_tool.py` | Core `BaseTool` class and decorators |
| `llm_base_tool.py` | Base class for LLM-powered tools |
| `internal_tools/` | Built-in tools (tool selector, parsers) |

## Usage

### Creating a Tool with Decorators

The simplest way to create a tool:

```python
from agents.tools_core.base_tool import create_fn_tool

@create_fn_tool(name="multiply", description="Multiply two numbers")
def multiply(x: int, y: int) -> int:
    return x * y

# Use it
result = multiply.invoke({"x": 5, "y": 3})
print(result.result)  # 15
```

### Creating a Tool with Classes

For more control, subclass `BaseTool`:

```python
from pydantic import BaseModel
from agents.tools_core.base_tool import BaseTool

class AddInput(BaseModel):
    a: float
    b: float

class AddOutput(BaseModel):
    result: float

class AddTool(BaseTool[AddInput, AddOutput]):
    _name = "add"
    description = "Add two numbers together"
    _input = AddInput
    _output = AddOutput

    def invoke(self, input: AddInput) -> AddOutput:
        return AddOutput(result=input.a + input.b)

# Use it
tool = AddTool()
result = tool.invoke(AddInput(a=2, b=3))
print(result.result)  # 5.0
```

### Async Tools

Override `ainvoke` for async operations:

```python
class AsyncSearchTool(BaseTool[SearchInput, SearchOutput]):
    _name = "search"
    description = "Search the web"
    _input = SearchInput
    _output = SearchOutput

    async def ainvoke(self, input: SearchInput) -> SearchOutput:
        result = await some_async_search(input.query)
        return SearchOutput(results=result)
```

### LLM-Powered Tools

For tools that use an LLM internally:

```python
from agents.tools_core.llm_base_tool import LLMTool

class SummarizerTool(LLMTool[SummaryInput, SummaryOutput]):
    _name = "summarize"
    description = "Summarize text using AI"
    _input = SummaryInput
    _output = SummaryOutput

    def format_messages(self, input: SummaryInput):
        return [
            {"role": "system", "content": "Summarize the following text:"},
            {"role": "user", "content": input.text}
        ]
```

## API Reference

### BaseTool

```python
class BaseTool[InputT, OutputT](Generic, ABC):
    _name: str              # Tool identifier
    description: str        # Description for LLM
    _input: type[BaseModel] # Input model class
    _output: type[BaseModel]# Output model class

    def invoke(input: InputT) -> OutputT    # Sync execution
    async def ainvoke(input: InputT) -> OutputT  # Async execution

    @property
    def name(self) -> str
    @property
    def input_model(self) -> type[BaseModel]
    @property
    def output_model(self) -> type[BaseModel]
```

### Decorators

```python
@create_fn_tool(name: str, description: str)
# Auto-generates input/output models from function signature

@create_tool(name: str, description: str, input_model: type, output_model: type)
# Explicit model specification
```

## Internal Tools

Located in `internal_tools/`:

| Tool | Purpose |
|------|---------|
| `ToolSelector` | LLM-based tool selection from a list |
| `ToolInputParser` | Parse natural language to tool input |
| `ToolOutputFormatter` | Format tool output for users |

## See Also

- [Agent Tool](../agent_tool/README.md) - How tools are executed
- [Tools](../tools/README.md) - Concrete tool implementations

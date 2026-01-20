# Agent Tool

Agent executor that orchestrates planning and tool execution.

## Concept

AgentTool runs an agentic loop:
1. Strategy decides what to do next
2. Tools are executed
3. Results added to message history
4. Repeat until task complete or max iterations

## Components

| File | Purpose |
|------|---------|
| `agent_tool.py` | Main `AgentTool` class |
| `planning_strategies.py` | Base class `PlanningStrategy` and `StrategyOutput` |
| `direct_strategy.py` | DirectStrategy - single LLM call |
| `react_strategy.py` | ReactStrategy - Reason-Act-Observe pattern |
| `adapt_strategy.py` | AdaptStrategy - try simple first, decompose on failure |
| `reflexion_strategy.py` | ReflexionStrategy - learn from mistakes through reflection |
| `adaptive_reflexion_strategy.py` | Combined ADaPT + Reflexion |
| `recursive/` | Sub-agent delegation support |
| `prompts/` | Jinja2 prompt templates |

## Usage

### Basic Agent

```python
import asyncio
from agents.agent_tool.agent_tool import AgentTool, AgentToolInput
from agents.agent_tool.direct_strategy import DirectStrategy
from agents.llm_core.llm_client import create_openai_client
from agents.tools.calculator import Calculator

async def main():
    client = create_openai_client()
    strategy = DirectStrategy(llm_client=client)

    agent = AgentTool(
        tools=[Calculator()],
        strategy=strategy,
    )

    result = await agent.ainvoke(
        AgentToolInput(objective="Calculate 25 * 4")
    )
    print(result.result)  # "100"
    print(result.success)  # True

asyncio.run(main())
```

### With Custom System Prompt

```python
agent = AgentTool(
    tools=[Calculator(), TavilySearch()],
    strategy=strategy,
    system_prompt="You are a research assistant. Be thorough.",
)
```

### With Max Iterations

```python
result = await agent.ainvoke(
    AgentToolInput(
        objective="Complex task...",
        max_iterations=10,  # Default is 5
    )
)
print(f"Used {result.iterations_used} iterations")
```

## Planning Strategies

### DirectStrategy

Single LLM call with tool calling. Fast and simple.

```python
from agents.agent_tool.direct_strategy import DirectStrategy

strategy = DirectStrategy(
    llm_client=client,
    model="gpt-4o-mini",  # Optional: override client's model
)
```

### ReactStrategy

Reason-Act-Observe pattern. Two-step process:
1. Generate reasoning about what to do
2. Select tool based on reasoning

```python
from agents.agent_tool.react_strategy import ReactStrategy

strategy = ReactStrategy(
    llm_client=client,
    model="gpt-4o",  # Optional: override client's model
)
)
```

### AdaptStrategy (ADaPT)

Try simple first, decompose on failure. Structural adaptation pattern:
1. Attempt direct solution
2. On failure, decompose into subtasks
3. Execute subtasks and combine results

```python
from agents.agent_tool.adapt_strategy import AdaptStrategy

strategy = AdaptStrategy(
    llm_client=client,
    max_direct_attempts=2,      # Attempts before decomposing
    error_threshold=0.5,        # Error rate to trigger decomposition
    stagnation_window=3,        # Iterations without progress
)
```

### ReflexionStrategy

Learn from mistakes through self-reflection. Behavioral adaptation pattern:
1. Attempt solution
2. On failure, reflect on what went wrong
3. Generate insights and retry with learned knowledge

```python
from agents.agent_tool.reflexion_strategy import ReflexionStrategy

strategy = ReflexionStrategy(
    llm_client=client,
    max_reflections=3,          # Max reflection cycles
    reflection_threshold=2,     # Failures before reflecting
)
```

### AdaptiveReflexionStrategy

Combined ADaPT + Reflexion for multi-layered recovery:
1. Attempt direct solution
2. On failure, reflect and retry (Reflexion)
3. If reflections exhausted, decompose (ADaPT)

```python
from agents.agent_tool.adaptive_reflexion_strategy import AdaptiveReflexionStrategy

strategy = AdaptiveReflexionStrategy(
    llm_client=client,
    max_reflections=2,
    max_direct_attempts=2,
)
```

## Recursive Agents

For complex hierarchical tasks, use sub-agents:

```python
from agents.agent_tool.recursive.runner import RecursiveAgentRunner
from agents.agent_tool.direct_strategy import DirectStrategy

runner = RecursiveAgentRunner(
    tools=[TavilySearch(), Calculator()],
    strategy_factory=lambda: DirectStrategy(llm_client),
    max_depth=3,
    max_iterations_per_level=5,
    include_sub_agent_at_root=True,
)

result = await runner.run(
    objective="Research and calculate...",
    context="Additional context",
)

# Access execution history
for level in result.execution_history:
    print(f"Depth {level.depth}: {level.objective}")
```

## API Reference

### AgentTool

```python
class AgentTool(BaseTool[AgentToolInput, AgentToolOutput]):
    def __init__(
        self,
        tools: list[BaseTool],
        strategy: PlanningStrategy,
        system_prompt: str | None = None,
        include_finish_tool: bool = True,
        parallel_tool_calls: bool = True,
    )
```

### AgentToolInput

```python
class AgentToolInput(BaseModel):
    objective: str           # Task to accomplish
    context: str = ""        # Additional context
    max_iterations: int = 5  # Max loop iterations
```

### AgentToolOutput

```python
class AgentToolOutput(BaseModel):
    result: str | None       # Final result
    success: bool            # Whether task succeeded
    iterations_used: int     # Number of iterations
    messages: list[...]      # Full message history
```

### PlanningStrategy

```python
class PlanningStrategy(ABC):
    async def plan(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool],
        parallel_tool_calls: bool = True,
    ) -> StrategyOutput
```

### StrategyOutput

```python
class StrategyOutput(BaseModel):
    messages: list[...]      # New messages (reasoning, etc.)
    tool_calls: list[ToolCall]  # Tools to execute
    finished: bool           # Task complete?
    success: bool            # Success status
    result: str | None       # Final result if finished
```

## Execution Flow

```
AgentToolInput.objective
        │
        ▼
┌───────────────────┐
│   Strategy.plan   │ ◄──────┐
└─────────┬─────────┘        │
          │                  │
          ▼                  │
    ┌─────────────┐          │
    │ Tool Calls? │──No──► Finished
    └─────┬───────┘          │
          │ Yes              │
          ▼                  │
┌───────────────────┐        │
│  Execute Tools    │        │
│  (parallel/seq)   │        │
└─────────┬─────────┘        │
          │                  │
          ▼                  │
┌───────────────────┐        │
│ Add to History    │────────┘
└───────────────────┘
```

## See Also

- [LLM Core](../llm_core/README.md) - LLM client used by strategies
- [Tools Core](../tools_core/README.md) - Tool base classes
- [Tools](../tools/README.md) - Available tools

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
| `base_strategy.py` | Base class `PlanningStrategy` and `StrategyOutput` |
| `direct_strategy.py` | DirectStrategy - single LLM call (plan-state-aware) |
| `react_strategy.py` | ReactStrategy - Reason-Act-Observe pattern |
| `adapt_strategy.py` | AdaptStrategy - try simple first, decompose on failure |
| `reflexion_strategy.py` | ReflexionStrategy - learn from mistakes through reflection |
| `adaptive_reflexion_strategy.py` | Combined ADaPT + Reflexion |
| `plan_state.py` | `PlanState` / `TaskState` - structured per-run plan |
| `meta_tool_plan_state_update.py` | `PlanStateUpdate` built-in meta tool that mutates `PlanState` |
| `meta_tool_finish.py` | `FinishInput`/`FinishOutput`/`create_finish_tool` — built-in meta tool that signals task completion |
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


**Execution Flow** 
```
Input → [LLM + Tools] → Tool Calls → Execute → Loop or Finish
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
```

**Execution Flow** 
```
Input → [Reason] → Thought → [Act] → Tool Call → Execute → [Observe] → Loop
            │                                                   │
            └──────────────── Reasoning Context ◄───────────────┘
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

**Execution Flow** 

```
Input → [Direct Attempt] ─success→ Finish
              │
           failure
              ▼
        [Decompose] → Subtask₁ → Subtask₂ → ... → [Combine] → Finish
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
**Execution Flow** 

```
Input → [Attempt] ─success→ Finish
             │
          failure
             ▼
       [Reflect] → Insights → [Retry with Memory] → Loop
             ▲                        │
             └────── if still failing ┘
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
**Execution Flow** 

```
Input → [Attempt] ─success→ Finish
             │
          failure
             ▼
       [Reflect] → [Retry] ─success→ Finish
             │          │
             │       failure (reflections exhausted)
             │          ▼
             └──► [Decompose] → Subtasks → [Combine] → Finish
```

## Meta tools

Two built-in tools are managed by `AgentTool` itself rather than the
strategy. They carry control signals (terminate, mutate plan state) and
follow a strict emission policy enforced in `_execute_tool_calls`.

| Meta tool | Purpose | Always added? | Configured via |
|---|---|---|---|
| `finish` | Signals task completion. Carries `result: str` and `success: bool`. AgentTool reads its args and exits the loop. | **Yes** — universal termination signal, no opt-out. | (always on) |
| `planstate_update` | Mutates the per-run `PlanState`. Provides full new task list and an optional `plan_status` (`completed`/`failed`/...). | Default yes. | `include_planstate_update_tool=True` (default) |

### Emission policy (HARD RULE — enforced by `_execute_tool_calls`)

Each meta tool MUST be emitted **alone** in its own iteration.
Forbidden combinations:

- `planstate_update` + any other tool (action, `finish`, or another `planstate_update`)
- `finish` + any other tool (including `planstate_update`)
- multiple meta tools in one turn

The framework rejects mixed turns: NO tool runs and the model receives a
policy-violation error result for each call. AgentTool's finish detection
also runs on the EXECUTED tool_calls, so a rejected `finish` cannot
short-circuit termination.

This guarantees `plan_state` cannot diverge from reality: a failed
`planstate_update` never co-exists with an action tool's side effect, and a
`finish` never terminates the loop with stale args from a never-recorded
plan revision.

### Termination paths

`AgentTool` finishes via exactly one of these per-iteration paths:

1. **PATH 1a — strategy-internal terminate**: `StrategyOutput.tool_calls == []`.
   The strategy decided to stop (typically because the LLM emitted no tool
   calls). Checked BEFORE tool execution. AgentTool reads `success`/`result`
   directly from `StrategyOutput`.
2. **PATH 1b — `finish` tool was called**: model emitted `finish` and the
   policy guard didn't reject. Checked AFTER tool execution. AgentTool
   extracts `result`/`success` from `finish`'s arguments.
3. **PATH 2 — terminal `plan_status`**: model called `planstate_update` with
   `plan_status='completed'` or `'failed'`. Checked after tool execution.
   AgentTool returns `success` based on whether the status is `completed`.
4. **PATH 3 — max iterations**: fallback. Returns `success=False`.

### Termination paths by configuration

Since `finish` is always available, the matrix only varies on
`include_planstate_update_tool`:

| `include_planstate_update_tool` | Available terminations |
|---|---|
| ✅ True (default) | 1a (no-tools), 1b (`finish`), 2 (`plan_status`), 3 (max_iter) |
| ❌ False | 1a (no-tools), 1b (`finish`), 3 (max_iter) — path 2 never fires (no tool to mutate `plan_status`) |

### Disabling `planstate_update`

Pass `include_planstate_update_tool=False` to opt out. The tool is added
per-run, so disabling it does not affect the static `agent.tools` list:

```python
agent = AgentTool(
    tools=[...],
    strategy=DirectStrategy(client),
    include_planstate_update_tool=False,  # default True
)
```

`PlanState` is still created (and returned via `AgentToolOutput.plan_state`),
but the model has no way to mutate it. Termination falls back to `finish`
or max_iterations.

## PlanState

Every `AgentTool.ainvoke()` creates a fresh `PlanState` for the run.
When `planstate_update` is enabled, the plan is re-injected into the
prompt as a `## Current Plan State` system block at every strategy call,
so the model reads its own plan as data instead of reconstructing it from
message history.

### Three-state, three-owner model

| State | Lifetime | Owner / mutator |
|---|---|---|
| `messages` (full history) | Durable per `ainvoke` call | `AgentTool` holds; appends `StrategyOutput.messages` delta after each turn |
| `PlanState` (plan + statuses) | Durable per `ainvoke` call | `AgentTool` initializes; `planstate_update` tool mutates (model-driven); `_execute_tool_calls` auto-updates statuses from execution |
| `StrategyOutput` (per-turn) | Transient — one iteration | Strategy produces; `AgentTool` consumes immediately |

### TaskState fields

- `id`, `objective`, `inputs`
- `status`: `pending | in_progress | completed | failed | blocked | cancelled`
- `result` (text or error)
- `depends_on`: list of task ids that must reach `completed` first
- `parent_attempt_id`: optional pointer to a prior failed attempt

### Auto-status-update

When exactly ONE non-meta tool call ran in a turn AND a task was
`in_progress` when the turn started, the framework auto-marks that task
`completed`/`failed` based on the tool's outcome and stores its output in
the task's `result` field. Parallel multi-action turns skip this auto-update
— the model must record outcomes via a follow-up `planstate_update`.

The framework does NOT auto-advance to the next task. After an action
auto-completes, `plan_state` has no `in_progress` task. The model is
expected to read the just-completed task's result, reason about it, and
then explicitly drive the next step via `planstate_update` or `finish`.

The model calls `planstate_update`:

- **(a)** to draft the initial plan (REQUIRED on iteration 1 when the tool is available),
- **(b)** to split/add tasks (per-item fan-out),
- **(c)** to re-plan when observations contradict the plan (retries, branches),
- **(d)** to record results for parallel multi-action batches (auto-update
  is skipped when more than one non-meta tool ran),
- **(e)** to terminate via `plan_status='completed' | 'failed'`,
- **(f)** to mark the next task `in_progress` after the previous one
  auto-completed (the framework does not advance for you).

### Per-item fan-out pattern

Each line below is one iteration. Meta tools always occupy a turn alone;
action tools occupy a turn alone (or as a parallel batch of independent
action tools).

```
1. planstate_update([                 ← (a) initial plan, meta alone
     {id:1, objective:"Fetch items", status:"in_progress"},
     {id:2, objective:"Process items", status:"pending", depends_on:[1]},
   ])
2. fetch_items()                      → ["a", "b", "c"]; auto-completes task 1
3. planstate_update([                 ← (b) split task 2 + (f) mark task 3 in_progress
     {id:1, ..., status:"completed"},
     {id:3, objective:"Process a", inputs:{item_id:"a"}, status:"in_progress",
            depends_on:[1]},
     {id:4, objective:"Process b", inputs:{item_id:"b"}, status:"pending",
            depends_on:[1]},
     {id:5, objective:"Process c", inputs:{item_id:"c"}, status:"pending",
            depends_on:[1]},
   ])
4. process_item(a)                    → auto-completes task 3
5. planstate_update(... task 4 in_progress)   ← (f)
6. process_item(b)                    → auto-completes task 4
7. planstate_update(... task 5 in_progress)   ← (f)
8. process_item(c)                    → auto-completes task 5
9. planstate_update(plan_status="completed")   ← (e) terminate via path 2
   OR  finish(success=True, result=...)
```

For independent items you can also dispatch them as a parallel batch —
`process_item(a)`, `process_item(b)`, `process_item(c)` together in one
turn. Auto-status-update is skipped for parallel turns, so the next
iteration must be a `planstate_update` (case **(d)**) to record per-task
outcomes.

### Sub-agent isolation

Because `PlanState` is created in `ainvoke` (not `__init__`), and `SubAgentTool`
constructs a fresh `AgentTool` per child invocation, each child gets its own
isolated plan. The parent's plan is invisible to the child and vice-versa.
`AgentToolOutput.plan_state` carries the final plan back from each invocation
for inspection.

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


**Execution Flow **
```
Objective → [Root d=0] → Tool? ─yes→ Execute → Result ──────────────────────────────→ Combine → Finish
                │                                                                       ▲
                └─ SubAgent? ─yes→ [Child d=1] → Tool? → Execute → Result ───Combine ───┘
                                        │                                       ▲
                                        └─ SubAgent? → [d=2] → ... → Result ────┘
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
        include_planstate_update_tool: bool = True,
        parallel_tool_calls: bool = True,
        guidance_messages: list[str] | None = None,
    )
```

`finish` is always added — there is no opt-out flag. See **Meta tools** above.

**Execution Flow**
```
Objective → [Strategy.plan] → Tool Calls? ─No→ Finished
                   ▲              │
                   │             Yes
                   │              ▼
                   └── History ← [Execute Tools]
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
    result: str | None             # Final result
    success: bool                  # Whether task succeeded
    iterations_used: int           # Number of iterations
    messages: list[...]            # Full message history
    plan_state: PlanState | None   # Final plan state for this run
```

### PlanningStrategy

```python
class PlanningStrategy(ABC):
    async def plan(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[BaseTool],
        parallel_tool_calls: bool = True,
        plan_state: PlanState | None = None,
    ) -> StrategyOutput
```

### StrategyOutput

```python
class StrategyOutput(BaseModel):
    messages: list[...]         # New messages (reasoning, etc.)
    tool_calls: list[ToolCall]  # Tools to execute (empty = strategy-internal terminate)
    success: bool = True        # Used ONLY when tool_calls == []
    result: str | None = None   # Used ONLY when tool_calls == []
```

`tool_calls` carries the termination signal implicitly:

- `tool_calls == []` → PATH 1a (strategy-internal terminate). `AgentTool`
  reads `success`/`result` from this `StrategyOutput`.
- a `finish` tool call in `tool_calls` → PATH 1b (after execution).
  `success`/`result` on `StrategyOutput` are ignored; they come from the
  `finish` tool's arguments.
- a `planstate_update` with terminal `plan_status` → PATH 2.

The strategy never special-cases `finish` itself — `AgentTool` owns that
detection.


## See Also

- [LLM Core](../llm_core/README.md) - LLM client used by strategies
- [Tools Core](../tools_core/README.md) - Tool base classes
- [Tools](../tools/README.md) - Available tools

# Agent Tool

Agent executor that orchestrates planning and tool execution.

## Concept

AgentTool runs an agentic loop:
1. Strategy decides what to do next
2. Tools are executed
3. Results added to message history
4. Repeat until task complete or max iterations

## Iteration anatomy

The full per-iteration shape, with explicit before/after of the
`strategy.plan()` call. Three pieces of state live across the loop and
have different ownership rules:

- **`messages`** — the durable conversation history. Owned by `AgentTool`;
  the strategy returns a per-turn delta that `AgentTool` appends.
- **`plan_state`** — the durable structured plan (tasks, statuses,
  results). Owned by `AgentTool`; mutated ONLY by the `planstate_update`
  tool (model-driven). The framework does NOT auto-mutate plan_state
  from tool results — the model reconciles every action batch via
  `planstate_update`. Strategies receive it by reference, read it, and
  generally do not mutate it directly (ReactStrategy's auto-translate
  step is the exception — it runs `planstate_update` programmatically).
- **`StrategyOutput`** — transient, per-turn. Strategy produces; AgentTool
  consumes immediately.

### Before `strategy.plan()` (once per `ainvoke`)

1. **Validate input** → `AgentToolInput { objective, context, max_iterations }`.
2. **Build per-run state**:
   - Create fresh `plan_state = PlanState(objective=...)` (durable for the
     whole run; sub-agents get their own).
   - Build `run_tools = self.tools + [PlanStateUpdate(plan_state)]` so the
     meta tool closes over *this* run's `plan_state`. `FINISH` is already
     in `self.tools` (always added; no opt-out).
3. **Seed `messages`**:
   - System prompt (from `agent_tool.jinja`; lists available tools).
   - Each `guidance_messages` entry as a system message.
   - User message with the rendered task prompt.
4. **Build `tool_map`**: `{tool.name.upper(): tool}` for case-insensitive dispatch.

### Per iteration (the loop body)

**Input** to `strategy.plan(...)`:

```python
strategy_output = await self.strategy.plan(
    messages=messages,            # full conversation history (read-only)
    tools=run_tools,              # action tools + FINISH + PLANSTATE_UPDATE
    parallel_tool_calls=...,      # bool
    plan_state=plan_state,        # by reference (read-only convention)
)
```

The strategy does whatever LLM calls it needs internally (DirectStrategy
= 1 call; ReactStrategy = 2–3 calls: reason → translate? → act). It does
NOT mutate `messages` directly — mutations come back through the return.

**Output** from `strategy.plan()`:

```python
StrategyOutput(
    messages: list[ChatCompletionMessageParam] = ...,   # per-turn delta
    tool_calls: list[ToolCall] = ...,                   # 0..N tools to run
    success: bool = True,                               # used ONLY when tool_calls == []
    result: str | None = None,                          # used ONLY when tool_calls == []
)
```

### After `strategy.plan()` (still per iteration)

1. **Apply message delta**: `messages.extend(strategy_output.messages)`.

2. **PATH 1a check** — strategy-internal terminate. If `tool_calls == []`:
   return `AgentToolOutput` with `success`/`result` from `StrategyOutput`,
   sync `plan_state.status` to terminal. Loop ends.

3. **Execute tool calls** via `_execute_tool_calls`:
   - **Policy guard**: allowed per-iteration combinations are
     (a) one `planstate_update` alone, (b) one `finish` alone, (c) one
     or more action tools, (d) `planstate_update` + `finish` together
     (Reconcile-and-finish mode). Anything else is rejected and the
     model gets a policy-violation error result.
   - **Phase A**: meta tools run first (sequentially). When both
     `planstate_update` and `finish` are emitted, `planstate_update` is
     sorted to run first so its mutations are visible before `finish`'s
     pre-termination housekeeping check. `planstate_update` mutates
     `plan_state` immediately. `finish`'s `acall` is a no-op
     acknowledgment; the real effect is in PATH 1b below.
   - **Phase B**: non-meta (action) tools run (parallel if
     `parallel_tool_calls=True`).
   - **Append tool-result messages** to `messages` in emission order
     (preserves OpenAI's `tool_call_id` pairing).
   - **No framework auto-update.** Action-tool results live in `messages`
     only; `plan_state` is NOT mutated by the framework. The model is
     responsible for reconciling `plan_state` against the new
     observations on the next iteration via `planstate_update`
     (Reconcile-and-plan / Reconcile-and-finish modes — see decision
     tree in the planner prompt).

4. **PATH 1b check** — `finish` was called. If any executed `tool_call`
   has `tool_name == "FINISH"`, the framework runs a
   **pre-termination housekeeping check** first: are any `plan_state`
   tasks still `pending`, `in_progress`, or `blocked`?
   - **If yes** (`plan_state` is not internally consistent): reject the
     FINISH. The framework rewrites the FINISH tool-result message in
     `messages` to a tool-error payload listing the offending task ids
     and explaining the housekeeping requirement. The loop **continues**
     to the next iteration. The model reads the error from message
     history and is expected to call `planstate_update` to mark each
     non-terminal task `completed` (with result from messages if the
     work was done), `cancelled` (if no longer reachable), or `failed`
     (if attempted and unrecoverable). Then re-emit `finish` on a
     subsequent iteration. Same shape as the parse-error / unknown-tool
     surfacing in `execute_single_tool`.
   - **If no** (all tasks terminal): extract `result` and `success`
     from `finish.arguments`, sync `plan_state.status` to `"completed"`
     / `"failed"`, return `AgentToolOutput`. Loop ends.

5. **PATH 2 check** — terminal `plan_status`. If `plan_state.status ∈
   {"completed", "failed"}` (set via `planstate_update`): return
   `AgentToolOutput` with `result = _summarize_plan_result(plan_state)`
   and `success = (status == "completed")`. Loop ends.

6. **Otherwise** → next iteration. `messages` and `plan_state` carry
   forward unchanged.

### After the loop ends

**PATH 3 — max iterations.** Return `AgentToolOutput(result="Max
iterations reached...", success=False, ...)`.

### Termination matrix

| Path | Trigger | Checked | `result` source | `plan_state.status` |
|---|---|---|---|---|
| 1a | `StrategyOutput.tool_calls == []` | Before tool execution | `strategy_output.result` (model text) or `_summarize_plan_result` fallback | Synced from `strategy_output.success` |
| 1b | `FINISH` tool call executed AND `plan_state` has no non-terminal tasks (housekeeping check passed) | After tool execution | `finish.arguments["result"]` (model-authored) | Synced from `finish.arguments["success"]` |
| 2 | `plan_status` ∈ `{completed, failed}` set via `planstate_update` | After tool execution | `_summarize_plan_result(plan_state)` (framework-derived) | Already terminal (model set it) |
| 3 | `iteration >= max_iterations` | After loop | Hardcoded `"Max iterations reached..."` | Untouched (typically `active`) |

**FINISH rejection (PATH 1b housekeeping)**: if `finish` is called while
`plan_state` has any task in `pending` / `in_progress` / `blocked`, PATH 1b
does NOT terminate. Instead, the framework rewrites the FINISH tool-result
message in `messages` to a tool-error payload listing the non-terminal
task ids, and the loop continues. The model is expected to call
`planstate_update` to mark each remaining task `completed` /
`cancelled` / `failed`, then re-emit `finish` on a later iteration.
The rejection only applies to PATH 1b — PATH 2 trusts the model's
explicit `plan_status` choice (the model just authored the plan).

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
        max_iterations=20,  # Default is 10
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

Reason-Act-Observe pattern with an optional plan-translation step.
Two or three LLM calls per iteration depending on configuration:

1. **Reasoning** — free-text analysis of state and prior tool results;
   identifies the IMMEDIATE next action.
2. **Plan translation** (optional, default ON via
   `auto_translate_plan=True` when `planstate_update` is among the
   tools): a focused LLM call takes the reasoning + current
   `plan_state` and emits a `planstate_update` mirroring the reasoning's
   intent (mark `in_progress`, add retries with `parent_attempt_id`,
   fan out, cancel obsolete branches). The strategy invokes the
   `planstate_update` meta tool to mutate `plan_state` in place. The
   action phase then sees the updated plan block.
3. **Action** — dispatches the tool(s) matching the current
   `in_progress` task(s). When auto-translate is active,
   `planstate_update` is removed from the action phase's tool list so
   the action LLM focuses on real work tools (+ `finish`).

```python
from agents.agent_tool.react_strategy import ReactStrategy

strategy = ReactStrategy(
    action_client=client,
    # Defaults: auto_translate_plan=True; translator client/model
    # default to the reasoning client/model.
)

# Or with auto-translate disabled (2 LLM calls per iter, model handles
# planstate_update itself):
strategy = ReactStrategy(
    action_client=client,
    auto_translate_plan=False,
)
```

**Execution flow with auto-translate (default)**
```
Input → [Reason] → free text → [Translate] → planstate_update → [Act] → Tool Call → Execute → Loop
                                    │
                                    └─► mutates plan_state (in_progress, retries, cancels)
```

Cost: 3 LLM calls per iteration. Benefit: `plan_state.tasks` stays a
faithful structured projection of the model's reasoning — parallel
batches, retries, replanning all encoded in the plan automatically.
Each `auto_translate` step also retries internally (capped by
`plan_translator_max_retries`, default 1) when the translator emits
malformed `planstate_update` args — same "surface error and let model
fix" pattern as `execute_single_tool`.

**Execution flow without auto-translate**
```
Input → [Reason] → free text → [Act] → Tool Call(s) → Execute → Loop
```
Cost: 2 LLM calls per iteration. Model is responsible for emitting
`planstate_update` itself when structural plan changes are needed.

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
| `planstate_update` | Mutates the per-run `PlanState`. Provides full new task list and an optional `plan_status` (`completed`/`failed`/...). | Default yes. | `enable_plan_state=True` (default) |

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

`AgentTool` finishes via exactly one of these per-iteration paths.
See **Iteration anatomy → Termination matrix** above for the full
per-path detail (when checked, `result` source, plan_status handling).

1. **PATH 1a — strategy-internal terminate**: `StrategyOutput.tool_calls == []`.
   The strategy decided to stop (typically because the LLM emitted no tool
   calls — a protocol violation since `finish` is always available).
   Checked BEFORE tool execution. AgentTool reads `success`/`result`
   directly from `StrategyOutput` and syncs `plan_state.status` to match.
2. **PATH 1b — `finish` tool was called**: model emitted `finish` and the
   policy guard didn't reject. Checked AFTER tool execution. Before
   honoring termination, the framework runs the **pre-termination
   housekeeping check** — if `plan_state` has any non-terminal tasks
   (`pending` / `in_progress` / `blocked`), FINISH is rejected by
   rewriting its tool-result message to a tool-error payload listing
   the offending task ids; the loop continues so the model can do the
   housekeeping via `planstate_update`. When the check passes, AgentTool
   extracts `result`/`success` from `finish`'s arguments and syncs
   `plan_state.status` to `"completed"` / `"failed"` to keep the
   returned `PlanState` internally consistent.
3. **PATH 2 — terminal `plan_status`**: model called `planstate_update` with
   `plan_status='completed'` or `'failed'`. Checked after tool execution.
   AgentTool returns `success` based on whether the status is `completed`;
   `result` is `_summarize_plan_result(plan_state)` (framework-derived).
4. **PATH 3 — max iterations**: fallback. Returns `success=False` with a
   hardcoded message.

### Termination paths by configuration

Since `finish` is always available, the matrix only varies on
`enable_plan_state`:

| `enable_plan_state` | Available terminations |
|---|---|
| ✅ True (default) | 1a (no-tools), 1b (`finish`), 2 (`plan_status`), 3 (max_iter) |
| ❌ False | 1a (no-tools), 1b (`finish`), 3 (max_iter) — path 2 never fires (no tool to mutate `plan_status`) |

### Disabling plan-state

Pass `enable_plan_state=False` to opt out. This skips auto-adding
`planstate_update` to the per-run tool list:

```python
agent = AgentTool(
    tools=[...],
    strategy=DirectStrategy(client),
    enable_plan_state=False,  # default True
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

### Plan-state lifecycle (model-driven reconciliation)

The framework does NOT mutate `plan_state` from tool results.
Every transition is driven by the model emitting `planstate_update`.

**Implication: after each action batch, `plan_state` is stale.** Tools
ran, results are in `messages`, but tasks still appear `in_progress`
until the model reconciles on the next iteration.

Per-iteration the model picks one of four modes (see the planner
prompt's decision tree for the full specification):

| Mode | When | Effect |
|---|---|---|
| **Reconcile-and-plan** | `plan_state` is diverged from reality AND objective not yet achieved | ONE `planstate_update` that records what just happened (Level 1+2 — status, `inputs`, `result`) AND plans the next step (Level 0 — mark next tasks `in_progress`) |
| **Reconcile-and-finish** | `plan_state` is diverged AND objective is achieved | `planstate_update` + `finish` together (framework runs planstate_update first); the planstate_update reconciles + marks remaining tasks terminal; finish terminates |
| **Plan** | `plan_state` is aligned AND objective not done | `planstate_update` for forward planning only (Level 0) |
| **Finish** | `plan_state` is aligned AND objective done | `finish` alone |

`task.inputs` and `task.result` are **retrospective audit fields**:
they're `None` for `pending`/`in_progress` tasks and populated during
reconciliation with the exact call args and tool output read from
message history. The pair is the canonical audit record of "we called
this tool with these args and got this back."

The framework does **NOT** auto-advance to the next task. After
reconciliation completes a task, the next `pending` task stays
`pending` until the model commits the transition via the same (or a
subsequent) `planstate_update`.

### Decision tree the model follows each iteration

```
1. Any task is `in_progress`?
   → Dispatch the action tool(s) matching those tasks
     (call args MUST dict-equal each task's inputs).
   ⚠️  Do NOT re-dispatch a tool that already ran — backfill via
       mode 2 instead. The framework does NOT gate dispatch: a
       non-matching, duplicate, or off-plan call STILL executes and
       you receive a `[plan_state hint]` describing the mismatch
       (one of: off-plan, duplicate, plan-needs-revision). Plan-state
       discipline is the only protection against duplicate
       non-idempotent calls.

2. Else, any pending task has all `depends_on` completed?
   → Call `planstate_update` to transition the plan:
     - Mark the next pending task in_progress, AND/OR
     - Split/add tasks (fan-out, retries, sub-steps), AND/OR
     - Cancel obsolete branches / rewire deps, AND/OR
     - Backfill from messages: if a tool ran successfully but its
       task is still pending/in_progress because the framework
       couldn't auto-record (inputs mismatch, partial overlap, or
       ambiguous match), copy the result from the conversation
       history into the task and mark it `completed`, AND/OR
     - Set `plan_status='completed'|'failed'` to terminate.

3. Else (no eligible work to do)
   → Pre-FINISH housekeeping FIRST: if `plan_state` has any task in
     `pending` / `in_progress` / `blocked`, you must call
     `planstate_update` to mark each terminal (`completed` /
     `cancelled` / `failed`) BEFORE calling `finish`. The framework
     enforces this — calling `finish` with non-terminal tasks
     surfaces a tool-error in the next iteration's message history
     telling you what to clean up.
   → Once `plan_state` is internally consistent: call `finish` with
     success=true (objective achieved) or success=false (no
     recoverable path remains).
```

The first iteration is a special case: the model emits the initial
`planstate_update` to draft the task list before any action runs. For
single-step objectives (one tool call then `finish`), `planstate_update`
can be skipped entirely.

**Handling `failed` and `blocked` tasks**: their dependents stay
`pending` (deps not `completed`). The model picks one of:

- **Retry** — add a new task with `parent_attempt_id` linking back,
  rewire dependents to depend on the new task.
- **Cancel** — mark the task `cancelled` and cascade-cancel any
  unreachable dependents.
- **Give up** — call `finish` with `success=false`.

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

### Iterating over tasks for display

`plan_state.tasks` is a flat list in **insertion order** (IDs stay stable
across re-plans, so insertion order can look scrambled when retries or
fan-outs add new IDs interleaved with the old ones). For a topologically
sorted view (by `depends_on`, with id-ascending as tiebreaker), use:

```python
for t in result.plan_state.tasks_in_display_order():
    print(f"[{t.id}] {t.status:<11} {t.objective}")
```

Same logic that the prompt's `## Current Plan State` block uses. Falls
back to id-order if a cycle or dangling `depends_on` is detected (the
run survives, just less prettily sorted).

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
        enable_plan_state: bool = True,
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
    objective: str               # Task to accomplish
    context: str | None = None   # Additional context
    max_iterations: int = 10     # Max loop iterations
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

"""
Context management for recursive agent execution.

Provides isolated context tracking via contextvars for:
- Recursion depth tracking
- Global execution history collection
- Statistics aggregation across all levels

Context isolation ensures each async execution chain has independent state,
while shared mutable structures (execution_history, statistics) allow
collecting information from all recursion levels.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
import typing as t


@dataclass
class LevelExecution:
    """
    Execution record for a single recursion level.

    Captures the complete state of one agent execution,
    including the full message history for debugging.
    """

    depth: int
    objective: str
    iterations_used: int
    success: bool
    result: str
    messages: list[dict[str, t.Any]] = field(default_factory=list)
    tool_calls_count: int = 0


@dataclass
class RecursionStatistics:
    """
    Aggregated statistics across all recursion levels.

    Updated as execution progresses through levels.
    """

    max_depth_reached: int = 0
    total_iterations: int = 0
    total_tool_calls: int = 0
    levels_completed: int = 0

    def update_from_level(
        self, depth: int, iterations: int, tool_calls: int = 0
    ) -> None:
        """Update statistics from a completed level execution."""
        self.max_depth_reached = max(self.max_depth_reached, depth)
        self.total_iterations += iterations
        self.total_tool_calls += tool_calls
        self.levels_completed += 1


# Current recursion depth (0 = root level)
var_recursion_depth: ContextVar[int] = ContextVar("recursion_depth", default=0)

# Maximum allowed recursion depth
var_max_depth: ContextVar[int] = ContextVar("max_depth", default=3)

# Current replan iteration within a level (for tracking replanning)
var_replan_iteration: ContextVar[int] = ContextVar("replan_iteration", default=0)

# Global execution history - collects LevelExecution from all levels
# Must be initialized before use (no default to prevent accidental sharing)
var_execution_history: ContextVar[list[LevelExecution]] = ContextVar(
    "execution_history"
)

# Global statistics - aggregates metrics across all levels
# Must be initialized before use
var_statistics: ContextVar[RecursionStatistics] = ContextVar("statistics")

# Parent objective (for context passing)
var_parent_objective: ContextVar[str] = ContextVar("parent_objective", default="")


def initialize_recursion_context(
    max_depth: int = 3,
) -> tuple[list[LevelExecution], RecursionStatistics]:
    """
    Initialize context variables for a new recursive execution.

    Must be called at the start of a recursive agent run.
    Returns references to the shared history and statistics for final collection.

    Args:
        max_depth: Maximum recursion depth allowed

    Returns:
        Tuple of (execution_history, statistics) for final collection
    """
    history: list[LevelExecution] = []
    stats = RecursionStatistics()

    var_recursion_depth.set(0)
    var_max_depth.set(max_depth)
    var_replan_iteration.set(0)
    var_execution_history.set(history)
    var_statistics.set(stats)
    var_parent_objective.set("")

    return history, stats


def get_current_depth() -> int:
    """Get current recursion depth (0 = root)."""
    return var_recursion_depth.get(0)


def get_max_depth() -> int:
    """Get maximum allowed recursion depth."""
    return var_max_depth.get(3)


def can_recurse() -> bool:
    """Check if further recursion is allowed."""
    return get_current_depth() < get_max_depth()


def increment_depth() -> int:
    """Increment depth and return new value."""
    new_depth = get_current_depth() + 1
    var_recursion_depth.set(new_depth)
    return new_depth


def record_level_execution(execution: LevelExecution) -> None:
    """
    Record a completed level execution to the global history.

    Also updates statistics automatically.
    """
    try:
        history = var_execution_history.get()
        history.append(execution)
    except LookupError:
        # Context not initialized - skip recording
        pass

    try:
        stats = var_statistics.get()
        stats.update_from_level(
            depth=execution.depth,
            iterations=execution.iterations_used,
            tool_calls=execution.tool_calls_count,
        )
    except LookupError:
        # Context not initialized - skip stats
        pass


def get_execution_history() -> list[LevelExecution]:
    """Get the global execution history (may raise LookupError if not initialized)."""
    return var_execution_history.get()


def get_statistics() -> RecursionStatistics:
    """Get the global statistics (may raise LookupError if not initialized)."""
    return var_statistics.get()

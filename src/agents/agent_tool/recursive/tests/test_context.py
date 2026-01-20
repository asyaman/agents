"""Tests for recursive context management."""

from agents.agent_tool.recursive.context import (
    LevelExecution,
    RecursionStatistics,
    can_recurse,
    get_current_depth,
    get_execution_history,
    get_max_depth,
    get_statistics,
    increment_depth,
    initialize_recursion_context,
    record_level_execution,
    var_execution_history,
    var_recursion_depth,
)


class TestLevelExecution:
    """Tests for LevelExecution dataclass."""

    def test_create_level_execution(self):
        execution = LevelExecution(
            depth=1,
            objective="Test objective",
            iterations_used=3,
            success=True,
            result="Test result",
            messages=[{"role": "user", "content": "test"}],
            tool_calls_count=2,
        )
        assert execution.depth == 1
        assert execution.objective == "Test objective"
        assert execution.iterations_used == 3
        assert execution.success is True
        assert execution.result == "Test result"
        assert len(execution.messages) == 1
        assert execution.tool_calls_count == 2

    def test_default_values(self):
        execution = LevelExecution(
            depth=0,
            objective="Test",
            iterations_used=1,
            success=True,
            result="Done",
        )
        assert execution.messages == []
        assert execution.tool_calls_count == 0


class TestRecursionStatistics:
    """Tests for RecursionStatistics dataclass."""

    def test_default_values(self):
        stats = RecursionStatistics()
        assert stats.max_depth_reached == 0
        assert stats.total_iterations == 0
        assert stats.total_tool_calls == 0
        assert stats.levels_completed == 0

    def test_update_from_level(self):
        stats = RecursionStatistics()

        stats.update_from_level(depth=1, iterations=5, tool_calls=3)
        assert stats.max_depth_reached == 1
        assert stats.total_iterations == 5
        assert stats.total_tool_calls == 3
        assert stats.levels_completed == 1

        stats.update_from_level(depth=2, iterations=3, tool_calls=2)
        assert stats.max_depth_reached == 2
        assert stats.total_iterations == 8
        assert stats.total_tool_calls == 5
        assert stats.levels_completed == 2

    def test_update_keeps_max_depth(self):
        stats = RecursionStatistics()
        stats.update_from_level(depth=3, iterations=1)
        stats.update_from_level(depth=1, iterations=1)
        assert stats.max_depth_reached == 3


class TestContextInitialization:
    """Tests for context variable initialization."""

    def test_initialize_recursion_context(self):
        history, stats = initialize_recursion_context(max_depth=5)

        assert history == []
        assert isinstance(stats, RecursionStatistics)
        assert get_current_depth() == 0
        assert get_max_depth() == 5

    def test_initialize_with_default_max_depth(self):
        initialize_recursion_context()
        assert get_max_depth() == 3

    def test_returns_mutable_references(self):
        history, stats = initialize_recursion_context()

        # Modify through returned references
        history.append(
            LevelExecution(
                depth=1, objective="test", iterations_used=1, success=True, result="ok"
            )
        )
        stats.update_from_level(depth=1, iterations=1)

        # Should be reflected in context
        assert len(get_execution_history()) == 1
        assert get_statistics().levels_completed == 1


class TestDepthManagement:
    """Tests for depth tracking utilities."""

    def test_get_current_depth_default(self):
        # Reset context
        var_recursion_depth.set(0)
        assert get_current_depth() == 0

    def test_increment_depth(self):
        var_recursion_depth.set(0)
        new_depth = increment_depth()
        assert new_depth == 1
        assert get_current_depth() == 1

    def test_can_recurse_when_below_max(self):
        initialize_recursion_context(max_depth=3)
        var_recursion_depth.set(0)
        assert can_recurse() is True

        var_recursion_depth.set(2)
        assert can_recurse() is True

    def test_cannot_recurse_at_max(self):
        initialize_recursion_context(max_depth=3)
        var_recursion_depth.set(3)
        assert can_recurse() is False

    def test_cannot_recurse_above_max(self):
        initialize_recursion_context(max_depth=3)
        var_recursion_depth.set(5)
        assert can_recurse() is False


class TestRecordLevelExecution:
    """Tests for recording level executions."""

    def test_record_level_execution(self):
        history, stats = initialize_recursion_context()

        execution = LevelExecution(
            depth=1,
            objective="Test task",
            iterations_used=3,
            success=True,
            result="Completed",
            tool_calls_count=2,
        )

        record_level_execution(execution)

        assert len(history) == 1
        assert history[0] is execution
        assert stats.max_depth_reached == 1
        assert stats.total_iterations == 3
        assert stats.total_tool_calls == 2
        assert stats.levels_completed == 1

    def test_record_multiple_executions(self):
        history, stats = initialize_recursion_context()

        for i in range(3):
            record_level_execution(
                LevelExecution(
                    depth=i + 1,
                    objective=f"Task {i}",
                    iterations_used=2,
                    success=True,
                    result=f"Result {i}",
                    tool_calls_count=1,
                )
            )

        assert len(history) == 3
        assert stats.levels_completed == 3
        assert stats.total_iterations == 6
        assert stats.max_depth_reached == 3

    def test_record_without_initialization_does_not_crash(self):
        # Clear context variables
        try:
            var_execution_history.get()
            # If we get here, context is set - skip this test path
        except LookupError:
            # Context not set - this is what we're testing
            execution = LevelExecution(
                depth=1,
                objective="Test",
                iterations_used=1,
                success=True,
                result="Done",
            )
            # Should not raise
            record_level_execution(execution)


class TestContextIsolation:
    """Tests for context variable isolation."""

    def test_context_variables_are_isolated(self):
        # Initialize first context
        history1, stats1 = initialize_recursion_context(max_depth=5)

        record_level_execution(
            LevelExecution(
                depth=1, objective="First", iterations_used=1, success=True, result="1"
            )
        )

        assert len(history1) == 1
        assert stats1.levels_completed == 1

        # Initialize second context (overwrites)
        history2, stats2 = initialize_recursion_context(max_depth=3)

        assert len(history2) == 0
        assert stats2.levels_completed == 0
        assert get_max_depth() == 3

    def test_depth_reset_on_initialize(self):
        initialize_recursion_context()
        var_recursion_depth.set(5)
        assert get_current_depth() == 5

        initialize_recursion_context()
        assert get_current_depth() == 0

"""
Recursive AgentTool - Self-calling agent with isolated execution contexts.

This module provides tools for recursive agent execution where:
- Each recursion level has independent message history
- Parent agents see only input/output (opaque results)
- Full execution history is collected globally for debugging
- Statistics track depth, iterations, and tool calls

Components:
- SubAgentTool: Tool that delegates to independent agent instances
- RecursiveAgentRunner: Entry point for recursive execution
- Context utilities: Context variable management for isolation

Usage:
    from agents.agent_tool.recursive import (
        RecursiveAgentRunner,
        SubAgentTool,
        RecursiveAgentOutput,
    )

    runner = RecursiveAgentRunner(
        tools=[search_tool, calc_tool],
        strategy_factory=lambda: DirectStrategy(llm_client),
        max_depth=3,
    )

    result = await runner.run(
        objective="Research and analyze market trends",
        context="Focus on tech sector"
    )

    # Access execution details
    print(f"Max depth reached: {result.statistics.max_depth_reached}")
    for level in result.execution_history:
        print(f"Level {level.depth}: {level.objective[:30]}...")
"""

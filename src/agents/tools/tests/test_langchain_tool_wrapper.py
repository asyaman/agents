"""
Tests for agents/tools/langchain_tool_wrapper.py

Tests:
- test_wrapper_invoke: Sync invocation works
- test_wrapper_ainvoke: Async invocation works
- test_wrapper_inherits_name_and_description: Name/description from langchain tool
- test_wrapper_uses_args_schema: Input schema from langchain tool
"""

import typing as t

import pytest
from langchain_core.tools import BaseTool as LangchainBaseTool
from pydantic import BaseModel

from agents.tools.langchain_tool_wrapper import (
    DefaultOutputSchema,
    LangchainToolWrapper,
)


class MockInput(BaseModel):
    query: str


class MockLangchainTool(LangchainBaseTool):
    name: str = "mock_tool"
    description: str = "A mock langchain tool"
    args_schema: t.Any = MockInput

    def _run(self, query: str) -> str:
        return f"Result: {query}"

    async def _arun(self, query: str) -> str:
        return f"Async Result: {query}"


@pytest.fixture
def mock_langchain_tool() -> MockLangchainTool:
    return MockLangchainTool()


@pytest.fixture
def wrapper(mock_langchain_tool: MockLangchainTool) -> LangchainToolWrapper:
    return LangchainToolWrapper(mock_langchain_tool)


class TestLangchainToolWrapper:
    def test_wrapper_inherits_name_and_description(self, wrapper: LangchainToolWrapper):
        assert wrapper.name == "MOCK_TOOL"
        assert wrapper.description == "A mock langchain tool"

    def test_wrapper_uses_args_schema(self, wrapper: LangchainToolWrapper):
        assert wrapper._input == MockInput

    def test_wrapper_invoke(self, wrapper: LangchainToolWrapper):
        result = wrapper.invoke(MockInput(query="test"))
        assert isinstance(result, DefaultOutputSchema)
        assert result.result == "Result: test"

    @pytest.mark.asyncio
    async def test_wrapper_ainvoke(self, wrapper: LangchainToolWrapper):
        result = await wrapper.ainvoke(MockInput(query="async test"))
        assert isinstance(result, DefaultOutputSchema)
        assert result.result == "Async Result: async test"

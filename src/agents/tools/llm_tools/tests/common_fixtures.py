"""Common fixtures for llm_tools tests."""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pydantic import BaseModel

from agents.tools_core.base_tool import BaseTool


class SimpleInput(BaseModel):
    query: str


class SimpleOutput(BaseModel):
    result: str


class SimpleTool(BaseTool[SimpleInput, SimpleOutput]):
    """Simple tool for testing wrappers."""

    _name = "simple_tool"
    description = "A simple test tool"
    _input = SimpleInput
    _output = SimpleOutput
    example_inputs = (SimpleInput(query="test query"),)
    example_outputs = (SimpleOutput(result="test result"),)

    def invoke(self, input: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=f"Result for: {input.query}")


@pytest.fixture
def simple_tool() -> SimpleTool:
    return SimpleTool()


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLMClient for testing LLM-based tools."""
    client = MagicMock()
    response = Mock()
    response.parsed = None  # Will be set per test
    client.generate.return_value = response
    client.agenerate = AsyncMock(return_value=response)
    return client

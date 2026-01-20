"""Common fixtures for agent_tool tests."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import BaseModel, Field

from agents.tools_core.base_tool import BaseTool


class SearchInput(BaseModel):
    """Input for search tool."""

    query: str = Field(description="Search query")


class SearchOutput(BaseModel):
    """Output from search tool."""

    results: list[str] = Field(description="Search results")


class SearchTool(BaseTool[SearchInput, SearchOutput]):
    """Simple search tool for testing."""

    _name = "search"
    description = "Search for information"
    _input = SearchInput
    _output = SearchOutput

    def invoke(self, input: SearchInput) -> SearchOutput:
        return SearchOutput(results=[f"Result for: {input.query}"])

    async def ainvoke(self, input: SearchInput) -> SearchOutput:
        return self.invoke(input)


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    expression: str = Field(description="Math expression to evaluate")


class CalculatorOutput(BaseModel):
    """Output from calculator tool."""

    result: float = Field(description="Calculation result")


class CalculatorTool(BaseTool[CalculatorInput, CalculatorOutput]):
    """Simple calculator tool for testing."""

    _name = "calculator"
    description = "Evaluate math expressions"
    _input = CalculatorInput
    _output = CalculatorOutput

    def invoke(self, input: CalculatorInput) -> CalculatorOutput:
        # Simple eval for testing (not safe for production!)
        result = eval(input.expression)  # noqa: S307
        return CalculatorOutput(result=float(result))

    async def ainvoke(self, input: CalculatorInput) -> CalculatorOutput:
        return self.invoke(input)


@pytest.fixture
def search_tool() -> SearchTool:
    """Create a search tool for testing."""
    return SearchTool()


@pytest.fixture
def calculator_tool() -> CalculatorTool:
    """Create a calculator tool for testing."""
    return CalculatorTool()


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock()
    client.agenerate = AsyncMock()
    client._default_model = "gpt-4"
    return client

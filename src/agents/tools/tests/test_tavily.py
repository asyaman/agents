"""
Tests for agents/tools/tavily.py

Tests:
- test_tavily_search_name_and_description: Name and description
- test_tavily_search_invoke: Invoke with mocked client
- test_tavily_search_schemas: Input/output schema structure
"""

from unittest.mock import MagicMock

import pytest

from agents.tools.tavily import TavilyInput, TavilyOutput, TavilySearch


@pytest.fixture
def mock_tavily_client() -> MagicMock:
    client = MagicMock()
    client.search.return_value = {
        "answer": "Test answer",
        "follow_up_questions": None,
        "response_time": 0.5,
        "query": "test query",
        "results": [
            {
                "content": "Test content",
                "raw_content": None,
                "score": 0.9,
                "title": "Test Title",
                "url": "https://example.com",
            }
        ],
    }
    return client


@pytest.fixture
def tavily_search(mock_tavily_client: MagicMock) -> TavilySearch:
    return TavilySearch(max_results=3, client=mock_tavily_client)


class TestTavilySearch:
    def test_name_and_description(self, tavily_search: TavilySearch):
        assert tavily_search.name == "TAVILY_SEARCH"
        assert "search" in tavily_search.description.lower()

    def test_invoke(self, tavily_search: TavilySearch, mock_tavily_client: MagicMock):
        result = tavily_search.invoke(TavilyInput(search_query="test query"))

        assert isinstance(result, TavilyOutput)
        assert result.query == "test query"
        assert result.answer == "Test answer"
        assert len(result.results) == 1
        assert result.results[0].title == "Test Title"
        mock_tavily_client.search.assert_called_once_with("test query", max_results=3)

    def test_input_output_schemas(self, tavily_search: TavilySearch):
        input_schema = tavily_search.input_schema()
        assert "search_query" in input_schema["properties"]

        output_schema = tavily_search.output_schema()
        assert "results" in output_schema["properties"]
        assert "query" in output_schema["properties"]

    def test_max_results_stored(self, mock_tavily_client: MagicMock):
        search = TavilySearch(max_results=5, client=mock_tavily_client)
        assert search.max_results == 5

"""Tests for ToolSelector - filters relevant tools based on user objective."""

from unittest.mock import MagicMock

import pytest

from agents.tools_core.internal_tools.tool_selector import (
    SelectedTool,
    ToolInfo,
    ToolSelector,
    ToolSelectorInput,
    ToolSelectorOutput,
)

# Default response for tool selector tests
_DEFAULT_RESPONSE = ToolSelectorOutput(
    selected_tools=[SelectedTool(name="web_search", reason="Needed for search")],
    reasoning="Selected web_search for the search task",
)


@pytest.fixture
def sample_tools() -> list[ToolInfo]:
    return [
        ToolInfo(
            name="web_search",
            description="Search the web",
            input_schema={"query": {"type": "string"}},
        ),
        ToolInfo(
            name="calculator",
            description="Perform calculations",
            input_schema={"expression": {"type": "string"}},
        ),
    ]


class TestToolSelector:
    def test_format_messages(
        self, mock_llm_client: MagicMock, sample_tools: list[ToolInfo]
    ):
        selector = ToolSelector(mock_llm_client)
        input_data = ToolSelectorInput(objective="Search for news", tools=sample_tools)

        messages = selector.format_messages(input_data)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Search for news" in messages[0]["content"]
        assert "web_search" in messages[0]["content"]

    def test_invoke(self, mock_llm_client: MagicMock, sample_tools: list[ToolInfo]):
        mock_llm_client.generate.return_value.parsed = _DEFAULT_RESPONSE
        selector = ToolSelector(mock_llm_client)
        input_data = ToolSelectorInput(objective="Search for news", tools=sample_tools)

        result = selector.invoke(input_data)

        assert len(result.selected_tools) == 1
        assert result.selected_tools[0].name == "web_search"
        mock_llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke(
        self, mock_llm_client: MagicMock, sample_tools: list[ToolInfo]
    ):
        mock_llm_client.agenerate.return_value.parsed = _DEFAULT_RESPONSE
        selector = ToolSelector(mock_llm_client)
        input_data = ToolSelectorInput(objective="Search for news", tools=sample_tools)

        result = await selector.ainvoke(input_data)

        assert len(result.selected_tools) == 1
        assert result.selected_tools[0].name == "web_search"
        mock_llm_client.agenerate.assert_called_once()

    def test_tools_to_info(self, simple_tool):
        infos = ToolSelector.tools_to_info([simple_tool])

        assert len(infos) == 1
        assert infos[0].name == simple_tool.name
        assert infos[0].description == simple_tool.description

    def test_select_from_tools(self, mock_llm_client: MagicMock, simple_tool):
        mock_llm_client.generate.return_value.parsed = _DEFAULT_RESPONSE
        selector = ToolSelector(mock_llm_client)

        result = selector.select_from_tools("test objective", [simple_tool])

        assert isinstance(result, ToolSelectorOutput)
        mock_llm_client.generate.assert_called_once()

    def test_filter_tools_returns_matching(
        self, mock_llm_client: MagicMock, simple_tool
    ):
        # Mock returns web_search, but we pass simple_tool - should return empty
        mock_llm_client.generate.return_value.parsed = _DEFAULT_RESPONSE
        selector = ToolSelector(mock_llm_client)

        filtered = selector.filter_tools("test", [simple_tool])

        assert filtered == []  # simple_tool name doesn't match "web_search"

    def test_filter_tools_with_match(self, mock_llm_client: MagicMock, simple_tool):
        mock_llm_client.generate.return_value.parsed = ToolSelectorOutput(
            selected_tools=[SelectedTool(name=simple_tool.name, reason="Needed")],
            reasoning="Selected for test",
        )
        selector = ToolSelector(mock_llm_client)

        filtered = selector.filter_tools("test", [simple_tool])

        assert len(filtered) == 1
        assert filtered[0].name == simple_tool.name

    def test_input_output_schemas(self, mock_llm_client: MagicMock):
        selector = ToolSelector(mock_llm_client)

        input_schema = selector.input_schema()
        assert "objective" in input_schema["properties"]
        assert "tools" in input_schema["properties"]

        output_schema = selector.output_schema()
        assert "selected_tools" in output_schema["properties"]
        assert "reasoning" in output_schema["properties"]

    def test_batch_tools_single_batch(self, mock_llm_client: MagicMock):
        selector = ToolSelector(mock_llm_client, batch_size=10)
        tools = [ToolInfo(name=f"tool_{i}", description=f"Tool {i}") for i in range(5)]

        batches = selector._batch_tools(tools)

        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_batch_tools_multiple_batches(self, mock_llm_client: MagicMock):
        selector = ToolSelector(mock_llm_client, batch_size=3)
        tools = [ToolInfo(name=f"tool_{i}", description=f"Tool {i}") for i in range(7)]

        batches = selector._batch_tools(tools)

        assert len(batches) == 3
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 1

    def test_merge_results_deduplicates(self, mock_llm_client: MagicMock):
        selector = ToolSelector(mock_llm_client)
        results = [
            ToolSelectorOutput(
                selected_tools=[SelectedTool(name="tool_a", reason="Reason 1")],
                reasoning="Batch 1",
            ),
            ToolSelectorOutput(
                selected_tools=[
                    SelectedTool(name="tool_a", reason="Reason 2"),
                    SelectedTool(name="tool_b", reason="Reason 3"),
                ],
                reasoning="Batch 2",
            ),
        ]

        merged = selector._merge_results(results)

        assert len(merged.selected_tools) == 2
        assert merged.selected_tools[0].name == "tool_a"
        assert merged.selected_tools[1].name == "tool_b"
        assert "Batch 1" in merged.reasoning
        assert "Batch 2" in merged.reasoning

    def test_select_with_batching(self, mock_llm_client: MagicMock, simple_tool):
        mock_llm_client.generate.return_value.parsed = _DEFAULT_RESPONSE
        selector = ToolSelector(mock_llm_client, batch_size=1)

        # Create 3 tools to trigger batching
        from agents.tools_core.base_tool import BaseTool
        from pydantic import BaseModel

        class DummyInput(BaseModel):
            x: str

        class DummyOutput(BaseModel):
            y: str

        class DummyTool(BaseTool[DummyInput, DummyOutput]):
            _name = "dummy"
            description = "Dummy tool"
            _input = DummyInput
            _output = DummyOutput

            def invoke(self, input: DummyInput) -> DummyOutput:
                return DummyOutput(y=input.x)

        tools = [DummyTool() for _ in range(3)]
        for i, t in enumerate(tools):
            t._name = f"tool_{i}"

        selector.select_from_tools("test", tools)

        # Should have called generate 3 times (once per batch)
        assert mock_llm_client.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_aselect_parallel_mode(self, mock_llm_client: MagicMock, simple_tool):
        mock_llm_client.agenerate.return_value.parsed = _DEFAULT_RESPONSE
        selector = ToolSelector(mock_llm_client, batch_size=1, parallel_mode=True)

        from agents.tools_core.base_tool import BaseTool
        from pydantic import BaseModel

        class DummyInput(BaseModel):
            x: str

        class DummyOutput(BaseModel):
            y: str

        class DummyTool(BaseTool[DummyInput, DummyOutput]):
            _name = "dummy"
            description = "Dummy tool"
            _input = DummyInput
            _output = DummyOutput

            def invoke(self, input: DummyInput) -> DummyOutput:
                return DummyOutput(y=input.x)

        tools = [DummyTool() for _ in range(3)]
        for i, t in enumerate(tools):
            t._name = f"tool_{i}"

        await selector.aselect_from_tools("test", tools)

        # Should have called agenerate 3 times (in parallel)
        assert mock_llm_client.agenerate.call_count == 3

    def test_include_input_schema_false(self, mock_llm_client: MagicMock):
        selector = ToolSelector(mock_llm_client, include_input_schema=False)
        tools = [ToolInfo(name="test", description="Test", input_schema={"x": "y"})]
        input_data = ToolSelectorInput(objective="test", tools=tools)

        messages = selector.format_messages(input_data)

        # Input schema should not appear in prompt when include_input_schema=False
        assert "Input Schema" not in messages[0]["content"]

    def test_tools_to_info_without_schema(self, simple_tool):
        infos = ToolSelector.tools_to_info([simple_tool], include_input_schema=False)

        assert len(infos) == 1
        assert infos[0].input_schema is None

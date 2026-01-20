"""
Tests for agents/tools/llm_tools/browsing.py

Tests:
- Input/Output model validation
- BrowsingTool initialization
- Agent loop with finish call
- Agent loop with max iterations
- Tool execution and error handling
- Navigation tracking
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.llm_core.llm_client import ToolCall, ToolCallResponse
from agents.tools.llm_tools.browsing import (
    BrowsingTool,
    BrowsingToolInput,
    BrowsingToolOutput,
    NavigateOutput,
    ClickOutput,
    GetTextOutput,
    GetPageInfoOutput,
    FinishOutput,
    create_browser_tools,
)


@pytest.fixture
def mock_page() -> MagicMock:
    """Mock Playwright page."""
    page = MagicMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example Page")
    page.goto = AsyncMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.wait_for_load_state = AsyncMock()

    # Mock locator
    locator = MagicMock()
    locator.inner_text = AsyncMock(return_value="Page content here")
    page.locator = MagicMock(return_value=locator)

    return page


@pytest.fixture
def mock_browser(mock_page: MagicMock) -> MagicMock:
    """Mock Playwright browser."""
    browser = MagicMock()
    browser.new_page = AsyncMock(return_value=mock_page)
    browser.close = AsyncMock()
    return browser


def make_tool_call_response(
    tool_name: str,
    arguments: dict,
    call_id: str = "call_123",
) -> ToolCallResponse:
    """Helper to create a ToolCallResponse."""
    return ToolCallResponse(
        tool_calls=[
            ToolCall(
                id=call_id,
                tool_name=tool_name,
                arguments=arguments,
            )
        ],
        finish_reason="tool_calls",
    )


class TestBrowsingToolInput:
    def test_default_values(self):
        input_data = BrowsingToolInput(query="Find something")
        assert input_data.query == "Find something"
        assert input_data.start_url is None
        assert input_data.max_iterations == 15

    def test_custom_values(self):
        input_data = BrowsingToolInput(
            query="Search for Python",
            start_url="https://google.com",
            max_iterations=10,
        )
        assert input_data.query == "Search for Python"
        assert input_data.start_url == "https://google.com"
        assert input_data.max_iterations == 10


class TestBrowsingToolOutput:
    def test_output_structure(self):
        output = BrowsingToolOutput(
            result="Found the answer",
            success=True,
            pages_visited=["https://example.com"],
            iterations_used=3,
        )
        assert output.result == "Found the answer"
        assert output.success is True
        assert output.pages_visited == ["https://example.com"]
        assert output.iterations_used == 3

    def test_default_pages_visited(self):
        output = BrowsingToolOutput(
            result="Result",
            success=True,
            iterations_used=1,
        )
        assert output.pages_visited == []


# ============================================================================
# BrowsingTool Initialization Tests
# ============================================================================


class TestBrowsingToolInit:
    def test_basic_init(self, mock_llm_client: MagicMock):
        tool = BrowsingTool(llm_client=mock_llm_client)
        assert tool.llm_client == mock_llm_client
        assert tool.model is None
        assert tool.headless is True

    def test_custom_init(self, mock_llm_client: MagicMock):
        tool = BrowsingTool(
            llm_client=mock_llm_client,
            model="gpt-4",
            headless=False,
        )
        assert tool.model == "gpt-4"
        assert tool.headless is False

    def test_tool_metadata(self, mock_llm_client: MagicMock):
        tool = BrowsingTool(llm_client=mock_llm_client)
        # Tool names are normalized to uppercase
        assert tool.name == "BROWSING_AGENT"
        assert "browsing" in tool.description.lower()
        assert tool._input == BrowsingToolInput
        assert tool._output == BrowsingToolOutput


# ============================================================================
# Browser Tools Tests
# ============================================================================


class TestCreateBrowserTools:
    def test_creates_all_tools(self, mock_page: MagicMock):
        tools = create_browser_tools(mock_page)
        tool_names = [t.name for t in tools]

        # Tool names are normalized to uppercase
        assert "NAVIGATE" in tool_names
        assert "CLICK" in tool_names
        assert "TYPE_TEXT" in tool_names
        assert "GET_TEXT" in tool_names
        assert "GET_PAGE_INFO" in tool_names
        assert "FINISH" in tool_names
        assert len(tools) == 6

    @pytest.mark.asyncio
    async def test_navigate_tool_success(self, mock_page: MagicMock):
        tools = create_browser_tools(mock_page)
        navigate_tool = next(t for t in tools if t.name == "NAVIGATE")

        result = await navigate_tool.acall({"url": "https://test.com"})

        mock_page.goto.assert_called_once()
        assert isinstance(result, NavigateOutput)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_navigate_tool_failure(self, mock_page: MagicMock):
        mock_page.goto = AsyncMock(side_effect=Exception("Network error"))
        tools = create_browser_tools(mock_page)
        navigate_tool = next(t for t in tools if t.name == "NAVIGATE")

        result = await navigate_tool.acall({"url": "https://test.com"})

        assert isinstance(result, NavigateOutput)
        assert result.success is False
        assert "Error" in result.title

    @pytest.mark.asyncio
    async def test_click_tool_success(self, mock_page: MagicMock):
        tools = create_browser_tools(mock_page)
        click_tool = next(t for t in tools if t.name == "CLICK")

        result = await click_tool.acall({"selector": "#button"})

        mock_page.click.assert_called_once_with("#button", timeout=5000)
        assert isinstance(result, ClickOutput)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_text_tool_success(self, mock_page: MagicMock):
        tools = create_browser_tools(mock_page)
        get_text_tool = next(t for t in tools if t.name == "GET_TEXT")

        result = await get_text_tool.acall({"selector": "body"})

        assert isinstance(result, GetTextOutput)
        assert result.success is True
        assert result.text == "Page content here"

    @pytest.mark.asyncio
    async def test_get_text_truncates_long_content(self, mock_page: MagicMock):
        long_text = "x" * 5000
        locator_mock = AsyncMock(return_value=long_text)
        mock_page.locator.return_value.inner_text = locator_mock
        tools = create_browser_tools(mock_page)
        get_text_tool = next(t for t in tools if t.name == "GET_TEXT")

        result = await get_text_tool.acall({"selector": "body"})

        assert len(result.text) < 5000
        assert "[truncated]" in result.text

    @pytest.mark.asyncio
    async def test_get_page_info_tool(self, mock_page: MagicMock):
        tools = create_browser_tools(mock_page)
        page_info_tool = next(t for t in tools if t.name == "GET_PAGE_INFO")

        result = await page_info_tool.acall({})

        assert isinstance(result, GetPageInfoOutput)
        assert result.url == "https://example.com"
        assert result.title == "Example Page"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_finish_tool(self, mock_page: MagicMock):
        tools = create_browser_tools(mock_page)
        finish_tool = next(t for t in tools if t.name == "FINISH")

        result = await finish_tool.acall({"result": "Done!", "success": True})

        assert isinstance(result, FinishOutput)
        assert result.acknowledged is True


# ============================================================================
# Agent Loop Tests
# ============================================================================


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_finish_on_first_iteration(
        self,
        mock_llm_client: MagicMock,
        mock_browser: MagicMock,
        mock_page: MagicMock,
    ):
        """Test agent completes when LLM calls finish."""
        mock_llm_client.agenerate.return_value = make_tool_call_response(
            tool_name="finish",
            arguments={"result": "Bitcoin price is $50,000", "success": True},
        )

        tool = BrowsingTool(llm_client=mock_llm_client)

        with patch(
            "agents.tools.llm_tools.browsing.async_playwright"
        ) as mock_playwright:
            mock_pw_instance = MagicMock()
            mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_playwright.return_value.__aenter__ = AsyncMock(
                return_value=mock_pw_instance
            )
            mock_playwright.return_value.__aexit__ = AsyncMock()

            result = await tool.ainvoke(
                BrowsingToolInput(query="Find Bitcoin price", max_iterations=5)
            )

        assert result.success is True
        assert result.result == "Bitcoin price is $50,000"
        assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_max_iterations_reached(
        self,
        mock_llm_client: MagicMock,
        mock_browser: MagicMock,
        mock_page: MagicMock,
    ):
        """Test agent stops at max iterations."""
        # Always return get_text, never finish
        mock_llm_client.agenerate.return_value = make_tool_call_response(
            tool_name="get_text",
            arguments={"selector": "body"},
        )

        tool = BrowsingTool(llm_client=mock_llm_client)

        with patch(
            "agents.tools.llm_tools.browsing.async_playwright"
        ) as mock_playwright:
            mock_pw_instance = MagicMock()
            mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_playwright.return_value.__aenter__ = AsyncMock(
                return_value=mock_pw_instance
            )
            mock_playwright.return_value.__aexit__ = AsyncMock()

            result = await tool.ainvoke(
                BrowsingToolInput(query="Find something", max_iterations=3)
            )

        assert result.success is False
        assert "Max iterations" in result.result
        assert result.iterations_used == 3
        assert mock_llm_client.agenerate.call_count == 3

    @pytest.mark.asyncio
    async def test_navigation_tracking(
        self,
        mock_llm_client: MagicMock,
        mock_browser: MagicMock,
        mock_page: MagicMock,
    ):
        """Test that navigated URLs are tracked."""
        # Tool names are normalized to UPPERCASE
        responses = [
            make_tool_call_response(
                tool_name="NAVIGATE",
                arguments={"url": "https://test.com"},
                call_id="call_1",
            ),
            make_tool_call_response(
                tool_name="FINISH",
                arguments={"result": "Done", "success": True},
                call_id="call_2",
            ),
        ]
        mock_llm_client.agenerate.side_effect = responses

        # Update mock_page.url after navigation
        mock_page.url = "https://test.com"

        tool = BrowsingTool(llm_client=mock_llm_client)

        with patch(
            "agents.tools.llm_tools.browsing.async_playwright"
        ) as mock_playwright:
            mock_pw_instance = MagicMock()
            mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_playwright.return_value.__aenter__ = AsyncMock(
                return_value=mock_pw_instance
            )
            mock_playwright.return_value.__aexit__ = AsyncMock()

            result = await tool.ainvoke(
                BrowsingToolInput(query="Navigate and finish", max_iterations=5)
            )

        assert result.success is True
        assert "https://test.com" in result.pages_visited

    @pytest.mark.asyncio
    async def test_start_url_navigation(
        self,
        mock_llm_client: MagicMock,
        mock_browser: MagicMock,
        mock_page: MagicMock,
    ):
        """Test that start_url is navigated to and tracked."""
        mock_llm_client.agenerate.return_value = make_tool_call_response(
            tool_name="finish",
            arguments={"result": "Done", "success": True},
        )

        tool = BrowsingTool(llm_client=mock_llm_client)

        with patch(
            "agents.tools.llm_tools.browsing.async_playwright"
        ) as mock_playwright:
            mock_pw_instance = MagicMock()
            mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_playwright.return_value.__aenter__ = AsyncMock(
                return_value=mock_pw_instance
            )
            mock_playwright.return_value.__aexit__ = AsyncMock()

            result = await tool.ainvoke(
                BrowsingToolInput(
                    query="Do something",
                    start_url="https://start.com",
                    max_iterations=5,
                )
            )

        mock_page.goto.assert_called_with(
            "https://start.com", wait_until="domcontentloaded"
        )
        assert "https://start.com" in result.pages_visited

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_failure(
        self,
        mock_llm_client: MagicMock,
        mock_browser: MagicMock,
        mock_page: MagicMock,
    ):
        """Test handling when LLM returns no tool calls."""
        # Return a non-ToolCallResponse
        mock_llm_client.agenerate.return_value = MagicMock(tool_calls=None)

        tool = BrowsingTool(llm_client=mock_llm_client)

        with patch(
            "agents.tools.llm_tools.browsing.async_playwright"
        ) as mock_playwright:
            mock_pw_instance = MagicMock()
            mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_playwright.return_value.__aenter__ = AsyncMock(
                return_value=mock_pw_instance
            )
            mock_playwright.return_value.__aexit__ = AsyncMock()

            result = await tool.ainvoke(
                BrowsingToolInput(query="Do something", max_iterations=5)
            )

        assert result.success is False
        assert "stopped without calling finish" in result.result

    @pytest.mark.asyncio
    async def test_tool_execution_error_handled(
        self,
        mock_llm_client: MagicMock,
        mock_browser: MagicMock,
        mock_page: MagicMock,
    ):
        """Test that tool execution errors are handled gracefully."""
        responses = [
            make_tool_call_response(
                tool_name="click",
                arguments={"selector": "#nonexistent"},
                call_id="call_1",
            ),
            make_tool_call_response(
                tool_name="finish",
                arguments={"result": "Handled error", "success": True},
                call_id="call_2",
            ),
        ]
        mock_llm_client.agenerate.side_effect = responses

        # Make click fail
        mock_page.click = AsyncMock(side_effect=Exception("Element not found"))

        tool = BrowsingTool(llm_client=mock_llm_client)

        with patch(
            "agents.tools.llm_tools.browsing.async_playwright"
        ) as mock_playwright:
            mock_pw_instance = MagicMock()
            mock_pw_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_playwright.return_value.__aenter__ = AsyncMock(
                return_value=mock_pw_instance
            )
            mock_playwright.return_value.__aexit__ = AsyncMock()

            result = await tool.ainvoke(
                BrowsingToolInput(query="Click something", max_iterations=5)
            )

        # Should still succeed because agent recovered
        assert result.success is True
        assert result.iterations_used == 2


# ============================================================================
# Schema Tests
# ============================================================================


class TestSchemas:
    def test_input_schema(self, mock_llm_client: MagicMock):
        tool = BrowsingTool(llm_client=mock_llm_client)
        schema = tool.input_schema()

        assert "query" in schema["properties"]
        assert "start_url" in schema["properties"]
        assert "max_iterations" in schema["properties"]

    def test_output_schema(self, mock_llm_client: MagicMock):
        tool = BrowsingTool(llm_client=mock_llm_client)
        schema = tool.output_schema()

        assert "result" in schema["properties"]
        assert "success" in schema["properties"]
        assert "pages_visited" in schema["properties"]
        assert "iterations_used" in schema["properties"]

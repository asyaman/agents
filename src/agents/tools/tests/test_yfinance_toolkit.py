"""
Tests for agents/tools/yfinance_toolkit.py

Tests:
- test_dict_to_string: Helper function
- test_create_yfinance_tool_creates_class: Factory creates tool class
- test_stock_price_tool: StockPriceTool structure
- test_company_info_tool: CompanyInfoTool structure
- test_historical_data_tool: HistoricalDataTool with period param
- test_all_yfinance_tools_list: ALL_YFINANCE_TOOLS completeness
"""

from unittest.mock import MagicMock, patch

from agents.tools.yfinance_toolkit import (
    ALL_YFINANCE_TOOLS,
    CompanyInfoTool,
    HistoricalDataInput,
    HistoricalDataTool,
    StockPriceTool,
    TickerInput,
    create_yfinance_tool,
    dict_to_string,
)


class TestHelpers:
    def test_dict_to_string(self):
        data = {"key1": "value1", "key2": "value2"}
        result = dict_to_string(data)
        assert "key1: value1" in result
        assert "key2: value2" in result
        assert "\n" in result


class TestCreateYfinanceTool:
    def test_creates_tool_class(self):
        ToolClass = create_yfinance_tool(
            name="test_tool",
            description="Test description",
            yf_attribute="info",
            example_output="test output",
        )

        assert ToolClass._name == "test_tool"
        assert ToolClass.description == "Test description"
        assert ToolClass._input == TickerInput
        assert len(ToolClass.example_inputs) > 0


class TestStockPriceTool:
    def test_name_and_description(self):
        tool = StockPriceTool()
        assert tool.name == "STOCK_PRICE"
        assert "price" in tool.description.lower()

    def test_input_output_schemas(self):
        tool = StockPriceTool()
        input_schema = tool.input_schema()
        assert "ticker" in input_schema["properties"]

        output_schema = tool.output_schema()
        assert "price" in output_schema["properties"]

    @patch("agents.tools.yfinance_toolkit.yfinance")
    def test_invoke(self, mock_yfinance: MagicMock):
        mock_ticker = MagicMock()
        mock_close = MagicMock()
        mock_close.iloc.__getitem__ = MagicMock(return_value=150.50)
        mock_ticker.history.return_value = {"Close": mock_close}
        mock_yfinance.Ticker.return_value = mock_ticker

        tool = StockPriceTool()
        result = tool.invoke(TickerInput(ticker="AAPL"))
        assert result.price == 150.50


class TestCompanyInfoTool:
    def test_name_and_description(self):
        tool = CompanyInfoTool()
        assert tool.name == "COMPANY_INFO"
        assert "info" in tool.description.lower()

    @patch("agents.tools.yfinance_toolkit.yfinance")
    def test_invoke(self, mock_yfinance: MagicMock):
        mock_ticker = MagicMock()
        mock_ticker.info = {"sector": "Technology", "industry": "Software"}
        mock_yfinance.Ticker.return_value = mock_ticker

        tool = CompanyInfoTool()
        result = tool.invoke(TickerInput(ticker="AAPL"))
        assert "Technology" in result.info
        assert "Software" in result.info


class TestHistoricalDataTool:
    def test_name_and_description(self):
        tool = HistoricalDataTool()
        assert tool.name == "HISTORICAL_DATA"

    def test_input_has_period(self):
        input_schema = HistoricalDataTool.input_schema()
        assert "ticker" in input_schema["properties"]
        assert "period" in input_schema["properties"]

    @patch("agents.tools.yfinance_toolkit.yfinance")
    def test_invoke(self, mock_yfinance: MagicMock):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value.to_csv.return_value = (
            "Date,Open,High,Low,Close\n2023-01-01,100,105,99,104"
        )
        mock_yfinance.Ticker.return_value = mock_ticker

        tool = HistoricalDataTool()
        result = tool.invoke(HistoricalDataInput(ticker="AAPL", period="1mo"))
        assert "Date" in result.data
        mock_ticker.history.assert_called_once_with(period="1mo")


class TestAllYfinanceTools:
    def test_list_not_empty(self):
        assert len(ALL_YFINANCE_TOOLS) > 0

    def test_all_are_tool_classes(self):
        for tool_cls in ALL_YFINANCE_TOOLS:
            assert hasattr(tool_cls, "_name")
            assert hasattr(tool_cls, "description")
            assert hasattr(tool_cls, "_input")
            assert hasattr(tool_cls, "_output")

    def test_contains_main_tools(self):
        tool_names = [t._name for t in ALL_YFINANCE_TOOLS]
        assert "stock_price" in tool_names
        assert "company_info" in tool_names
        assert "historical_data" in tool_names

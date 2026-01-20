"""YFinance toolkit - tools for fetching stock market data."""

import json
import typing as t

import yfinance  # type: ignore
from pydantic import BaseModel, Field

from agents.tools_core.base_tool import BaseTool


def dict_to_string(data: t.Dict[t.Any, t.Any]) -> str:
    """Convert a dictionary to a newline-separated string."""
    return "\n".join([f"{key}: {value}" for key, value in data.items()])


class TickerInput(BaseModel):
    """Standard input for ticker-only tools."""

    ticker: str = Field(description="The stock ticker symbol.")


class StringOutput(BaseModel):
    """Standard output for string data."""

    data: str = Field(description="The result data as a string.")


def create_yfinance_tool(
    name: str,
    description: str,
    yf_attribute: str,
    to_dict_args: dict[str, t.Any] | None = None,
    example_ticker: str = "MSFT",
    example_output: str = "Example data",
) -> type[BaseTool[TickerInput, StringOutput]]:
    """
    Factory to create simple yfinance tools that follow the pattern:
    ticker → yfinance.Ticker(ticker).{attribute}.to_dict() → string
    """
    to_dict_kwargs = to_dict_args or {}

    class YFinanceTool(BaseTool[TickerInput, StringOutput]):
        _name = name
        _input = TickerInput
        _output = StringOutput

        def invoke(self, input: TickerInput) -> StringOutput:
            stock = yfinance.Ticker(input.ticker)
            attr = getattr(stock, yf_attribute)
            # Handle both property and method attributes
            data = attr.to_dict(**to_dict_kwargs) if hasattr(attr, "to_dict") else attr
            return StringOutput(data=dict_to_string(data))  # type: ignore

        example_inputs = (TickerInput(ticker=example_ticker),)
        example_outputs = (StringOutput(data=example_output),)

    YFinanceTool.description = description
    YFinanceTool.__name__ = f"{name.title().replace('_', '')}Tool"
    return YFinanceTool


# Tool definitions: (name, description, yf_attribute, to_dict_args, example_output)
_SIMPLE_TOOL_CONFIGS: list[tuple[str, str, str, dict[str, t.Any] | None, str]] = [
    (
        "dividends",
        "Fetch dividends for a stock ticker.",
        "dividends",
        None,
        "2023-12-31: 0.56",
    ),
    ("splits", "Fetch stock splits for a ticker.", "splits", None, "2023-12-31: 2"),
    (
        "capital_gains",
        "Fetch capital gains for a ticker.",
        "capital_gains",
        None,
        "2023-12-31: 1.2",
    ),
    (
        "balance_sheet",
        "Fetch balance sheet for a ticker.",
        "balance_sheet",
        None,
        "2023-12-31: {...}",
    ),
    (
        "income_statement",
        "Fetch income statement for a ticker.",
        "financials",
        None,
        "2023-12-31: {...}",
    ),
    (
        "cash_flow",
        "Fetch cash flow statement for a ticker.",
        "cashflow",
        None,
        "2023-12-31: {...}",
    ),
    (
        "major_holders",
        "Fetch major holders for a ticker.",
        "major_holders",
        {"orient": "index"},
        "Name: {...}",
    ),
    (
        "institutional_holders",
        "Fetch institutional holders.",
        "institutional_holders",
        None,
        "Institution: {...}",
    ),
    (
        "mutual_fund_holders",
        "Fetch mutual fund holders.",
        "mutualfund_holders",
        None,
        "Fund: {...}",
    ),
    (
        "insider_transactions",
        "Fetch insider transactions.",
        "insider_transactions",
        None,
        "Transaction: {...}",
    ),
    (
        "insider_purchases",
        "Fetch insider purchases.",
        "insider_purchases",
        None,
        "Purchase: {...}",
    ),
    (
        "insider_roster_holders",
        "Fetch insider roster holders.",
        "insider_roster_holders",
        None,
        "Holder: {...}",
    ),
    (
        "sustainability",
        "Fetch sustainability/ESG data.",
        "sustainability",
        None,
        "environmentScore: 10",
    ),
    (
        "recommendations",
        "Fetch analyst recommendations.",
        "recommendations",
        None,
        "firm: Goldman Sachs",
    ),
    (
        "recommendations_summary",
        "Fetch recommendations summary.",
        "recommendations_summary",
        None,
        "buy: 10",
    ),
    (
        "upgrades_downgrades",
        "Fetch upgrades and downgrades.",
        "upgrades_downgrades",
        None,
        "rating: Upgrade",
    ),
    (
        "earnings_dates",
        "Fetch earnings dates.",
        "earnings_dates",
        None,
        "date: 2023-08-01",
    ),
]

# Generate tool classes
DividendsTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[0])
SplitsTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[1])
CapitalGainsTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[2])
BalanceSheetTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[3])
IncomeStatementTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[4])
CashFlowTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[5])
MajorHoldersTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[6])
InstitutionalHoldersTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[7])
MutualFundHoldersTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[8])
InsiderTransactionsTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[9])
InsiderPurchasesTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[10])
InsiderRosterHoldersTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[11])
SustainabilityTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[12])
RecommendationsTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[13])
RecommendationsSummaryTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[14])
UpgradesDowngradesTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[15])
EarningsDatesTool = create_yfinance_tool(*_SIMPLE_TOOL_CONFIGS[16])


# Explicit Tools (Unique Logic)
# =============================================================================


class CompanyInfoOutput(BaseModel):
    info: str = Field(description="The company info as a JSON string.")


class CompanyInfoTool(BaseTool[TickerInput, CompanyInfoOutput]):
    """Fetch company info - uses JSON serialization instead of dict_to_string."""

    _name = "company_info"
    description = "Fetch company info for a given ticker symbol."
    _input = TickerInput
    _output = CompanyInfoOutput

    def invoke(self, input: TickerInput) -> CompanyInfoOutput:
        stock = yfinance.Ticker(input.ticker)
        info: t.Dict[str, t.Any] = stock.info  # type: ignore
        return CompanyInfoOutput(info=json.dumps(info))

    example_inputs = (TickerInput(ticker="AAPL"),)
    example_outputs = (CompanyInfoOutput(info='{"sector": "Technology", ...}'),)


# =============================================================================


class StockPriceOutput(BaseModel):
    price: float = Field(description="The current stock price.")


class StockPriceTool(BaseTool[TickerInput, StockPriceOutput]):
    """Fetch current stock price - returns float, not string."""

    _name = "stock_price"
    description = "Fetch the current stock price for a given ticker symbol."
    _input = TickerInput
    _output = StockPriceOutput

    def invoke(self, input: TickerInput) -> StockPriceOutput:
        stock = yfinance.Ticker(input.ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]  # type: ignore
        return StockPriceOutput(price=price)  # type: ignore

    example_inputs = (TickerInput(ticker="AAPL"),)
    example_outputs = (StockPriceOutput(price=150.25),)


# =============================================================================


class HistoricalDataInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol.")
    period: str = Field(
        description="The period to fetch (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')."
    )


class HistoricalDataOutput(BaseModel):
    data: str = Field(description="The historical stock data as CSV.")


class HistoricalDataTool(BaseTool[HistoricalDataInput, HistoricalDataOutput]):
    """Fetch historical data - has period parameter."""

    _name = "historical_data"
    description = "Fetch historical stock data for a given ticker symbol and period."
    _input = HistoricalDataInput
    _output = HistoricalDataOutput

    def invoke(self, input: HistoricalDataInput) -> HistoricalDataOutput:
        stock = yfinance.Ticker(input.ticker)
        data = stock.history(period=input.period).to_csv()  # type: ignore
        return HistoricalDataOutput(data=data)

    example_inputs = (HistoricalDataInput(ticker="AAPL", period="1mo"),)
    example_outputs = (HistoricalDataOutput(data="Date,Open,High,Low,Close,..."),)


# =============================================================================


class ShareCountInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol.")
    start: str = Field(description="Start date (YYYY-MM-DD).")
    end: str = Field(description="End date (YYYY-MM-DD).")


class ShareCountTool(BaseTool[ShareCountInput, StringOutput]):
    """Fetch share count - has start/end date parameters."""

    _name = "share_count"
    description = "Fetch share count for a ticker between specified dates."
    _input = ShareCountInput
    _output = StringOutput

    def invoke(self, input: ShareCountInput) -> StringOutput:
        stock = yfinance.Ticker(input.ticker)
        data = stock.get_shares_full(start=input.start, end=input.end).to_dict()  # type: ignore
        return StringOutput(data=dict_to_string(data))  # type: ignore

    example_inputs = (
        ShareCountInput(ticker="AAPL", start="2022-01-01", end="2022-12-31"),
    )
    example_outputs = (StringOutput(data="2022-01-01: 1000000000"),)


# =============================================================================


class ISINOutput(BaseModel):
    isin: str = Field(description="The ISIN code.")


class ISINTool(BaseTool[TickerInput, ISINOutput]):
    """Fetch ISIN code - returns string directly, no conversion."""

    _name = "isin"
    description = "Fetch the ISIN code for a given stock ticker symbol."
    _input = TickerInput
    _output = ISINOutput

    def invoke(self, input: TickerInput) -> ISINOutput:
        stock = yfinance.Ticker(input.ticker)
        return ISINOutput(isin=stock.isin)  # type: ignore

    example_inputs = (TickerInput(ticker="MSFT"),)
    example_outputs = (ISINOutput(isin="US5949181045"),)


# =============================================================================


class OptionsExpirationsOutput(BaseModel):
    options: str = Field(description="The options expiration dates.")


class OptionsExpirationsTool(BaseTool[TickerInput, OptionsExpirationsOutput]):
    """Fetch options expirations - returns tuple, needs join."""

    _name = "options_expirations"
    description = "Fetch options expiration dates for a given stock ticker."
    _input = TickerInput
    _output = OptionsExpirationsOutput

    def invoke(self, input: TickerInput) -> OptionsExpirationsOutput:
        stock = yfinance.Ticker(input.ticker)
        options = stock.options  # type: ignore
        return OptionsExpirationsOutput(options="\n".join(options))  # type: ignore

    example_inputs = (TickerInput(ticker="MSFT"),)
    example_outputs = (OptionsExpirationsOutput(options="2023-08-01\n2023-09-01"),)


# =============================================================================


class NewsOutput(BaseModel):
    news: list[dict[str, t.Any]] = Field(description="News articles.")


class NewsTool(BaseTool[TickerInput, NewsOutput]):
    """Fetch news - returns list of dicts, not string."""

    _name = "stock_news"
    description = "Fetch news articles for a given ticker symbol."
    _input = TickerInput
    _output = NewsOutput

    def invoke(self, input: TickerInput) -> NewsOutput:
        stock = yfinance.Ticker(input.ticker)
        return NewsOutput(news=stock.news)  # type: ignore

    example_inputs = (TickerInput(ticker="MSFT"),)
    example_outputs = (NewsOutput(news=[{"title": "Example", "link": "https://..."}]),)


# =============================================================================


ALL_YFINANCE_TOOLS: list[type[BaseTool[t.Any, t.Any]]] = [
    CompanyInfoTool,
    StockPriceTool,
    HistoricalDataTool,
    DividendsTool,
    SplitsTool,
    CapitalGainsTool,
    ShareCountTool,
    BalanceSheetTool,
    IncomeStatementTool,
    CashFlowTool,
    MajorHoldersTool,
    InstitutionalHoldersTool,
    MutualFundHoldersTool,
    InsiderTransactionsTool,
    InsiderPurchasesTool,
    InsiderRosterHoldersTool,
    SustainabilityTool,
    RecommendationsTool,
    RecommendationsSummaryTool,
    UpgradesDowngradesTool,
    EarningsDatesTool,
    ISINTool,
    OptionsExpirationsTool,
    NewsTool,
]

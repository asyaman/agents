# Tools

Concrete tool implementations for agents.

## Available Tools

| Tool | File | Description |
|------|------|-------------|
| Calculator | `calculator.py` | Mathematical expression evaluation |
| TavilySearch | `tavily.py` | Web search via Tavily API |
| YFinance | `yfinance_toolkit.py` | Financial data (stocks, prices) |
| Summarizer | `llm_tools/summarizer.py` | LLM-based text summarization |
| Browsing | `llm_tools/browsing.py` | Playwright-based web browsing |
| LLMCall | `llm_tools/llmcall.py` | Generic LLM calling tool |

## Usage Examples

### Calculator

Evaluates mathematical expressions using `numexpr`.

```python
from agents.tools.calculator import Calculator

calc = Calculator()
result = calc.invoke({"expression": "2 + 3 * 4"})
print(result.result)  # 14.0

# Supports complex expressions
result = calc.invoke({"expression": "(15 ** 2) / 3 + sqrt(16)"})
print(result.result)  # 79.0
```

### TavilySearch

Web search with follow-up questions.

```python
from agents.tools.tavily import TavilySearch

search = TavilySearch()
result = await search.ainvoke({"search_query": "Python asyncio tutorial"})

print(result.answer)
print(result.results)  # List of search results
print(result.follow_up_questions)
```

### YFinance

Get stock prices and company information.

```python
from agents.tools.yfinance_toolkit import (
    GetStockPrice,
    GetCompanyInfo,
    GetHistoricalPrices,
)

# Current price
price_tool = GetStockPrice()
result = price_tool.invoke({"ticker": "AAPL"})
print(result.price)

# Company info
info_tool = GetCompanyInfo()
result = info_tool.invoke({"ticker": "MSFT"})
print(result.company_name)
print(result.market_cap)
```

### Summarizer

LLM-powered text summarization.

```python
from agents.tools.llm_tools.summarizer import Summarizer
from agents.llm_core.llm_client import create_openai_client

client = create_openai_client()
summarizer = Summarizer(llm_client=client)

result = await summarizer.ainvoke({
    "text": "Long article text here...",
    "max_length": 100,
})
print(result.summary)
```

### Browsing

Web page content extraction with Playwright.

```python
from agents.tools.llm_tools.browsing import BrowsingTool
from agents.llm_core.llm_client import create_openai_client

client = create_openai_client()
browser = BrowsingTool(llm_client=client)

result = await browser.ainvoke({
    "url": "https://example.com",
    "objective": "Extract main content",
})
print(result.content)
```

## Creating Custom Tools

See [Tools Core](../tools_core/README.md) for how to create your own tools.

Quick example:

```python
from agents.tools_core.base_tool import create_fn_tool

@create_fn_tool(name="greet", description="Generate a greeting")
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

## Environment Variables

Some tools require API keys:

```env
# Tavily (web search)
TAVILY_API_KEY=tvly-...

```

## See Also

- [Tools Core](../tools_core/README.md) - Base classes for building tools
- [Agent Tool](../agent_tool/README.md) - How tools are used by agents

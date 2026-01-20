import typing as t

from pydantic import BaseModel, Field
from tavily import TavilyClient  # type: ignore

from agents.settings import get_settings
from agents.tools_core.base_tool import BaseTool


def default_tavily_client() -> TavilyClient:
    settings = get_settings()
    return TavilyClient(api_key=settings.tavily_api_key)


class TavilyResults(BaseModel):
    content: str = Field(description="Content from the search results.")
    raw_content: str | None = Field(description="Raw content from the search results.")
    score: float = Field(description="How related the result is to the query.")
    title: str = Field(description="Title of the webpage.")
    url: str = Field(description="Url of the webpage.")


class TavilyOutput(BaseModel):
    answer: str | None = Field(description="Answer to the question.")
    follow_up_questions: list[str] | None = Field(
        description="Any follow up questions that could help answer the query."
    )
    response_time: float = Field(description="How long it took to execute.")
    query: str = Field(description="The input query.")
    results: list[TavilyResults] = Field(description="The search results.")


class TavilyInput(BaseModel):
    search_query: str = Field(description="Query to search the web.")


class TavilySearch(BaseTool[TavilyInput, TavilyOutput]):
    _name = "tavily_search"
    description = "Search the web. Will return most related web pages and the content."
    _input = TavilyInput
    _output = TavilyOutput

    def __init__(
        self, max_results: int = 1, client: TavilyClient | None = None
    ) -> None:
        self.max_results = max_results
        self.client = client or default_tavily_client()
        super().__init__()

    def invoke(self, input: TavilyInput) -> TavilyOutput:
        result: dict[str, t.Any] = self.client.search(  # type: ignore
            input.search_query, max_results=self.max_results
        )
        return TavilyOutput(**result)

    example_inputs = (TavilyInput(search_query="..."),)
    example_outputs = (
        TavilyOutput(
            answer=None,
            follow_up_questions=None,
            response_time=0.5,
            query="Example query.",
            results=[
                TavilyResults(
                    content="Website content.",
                    raw_content=None,
                    score=0.4,
                    title="Website title",
                    url="www.example.com",
                )
            ],
        ),
    )

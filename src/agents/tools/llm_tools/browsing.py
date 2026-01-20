"""
Browsing agent tool using LLMClient and Playwright.

This tool creates an agentic loop that uses an LLM to control a browser
via Playwright. It supports any provider through LLMClient (OpenAI, Ollama, etc.).
"""

import asyncio
from loguru import logger
from playwright.async_api import Page, async_playwright
from pydantic import BaseModel, Field

from openai.types.chat import ChatCompletionMessageParam

from agents.llm_core.llm_client import LLMClient, ToolCallResponse
from agents.tools_core.base_tool import BaseTool, create_fn_tool


class BrowsingToolInput(BaseModel):
    """Input for the browsing agent."""

    query: str = Field(description="The task to accomplish using the browser.")
    start_url: str | None = Field(
        default=None,
        description="Optional starting URL. If not provided, agent decides where to navigate.",
    )
    max_iterations: int = Field(
        default=15,
        description="Maximum number of agent iterations before stopping.",
    )


class BrowsingToolOutput(BaseModel):
    """Output from the browsing agent."""

    result: str = Field(description="The result of the browsing task.")
    success: bool = Field(description="Whether the task was completed successfully.")
    pages_visited: list[str] = Field(
        default_factory=list, description="List of URLs visited during the task."
    )
    iterations_used: int = Field(description="Number of iterations used.")


# Browser Action Tools (used by the agent internally)
class NavigateInput(BaseModel):
    url: str = Field(description="The URL to navigate to.")


class NavigateOutput(BaseModel):
    success: bool
    current_url: str
    title: str


class ClickInput(BaseModel):
    selector: str = Field(
        description="CSS selector of the element to click (e.g., 'button#submit', 'a.link')."
    )


class ClickOutput(BaseModel):
    success: bool
    message: str


class TypeTextInput(BaseModel):
    selector: str = Field(description="CSS selector of the input element.")
    text: str = Field(description="Text to type into the element.")


class TypeTextOutput(BaseModel):
    success: bool
    message: str


class GetTextInput(BaseModel):
    selector: str = Field(
        default="body",
        description="CSS selector to extract text from. Defaults to 'body' for full page.",
    )


class GetTextOutput(BaseModel):
    text: str
    success: bool


class GetPageInfoInput(BaseModel):
    """No input needed - gets info about current page."""

    pass


class GetPageInfoOutput(BaseModel):
    url: str
    title: str
    success: bool


class FinishInput(BaseModel):
    """Signal that the task is complete."""

    result: str = Field(description="The final result/answer for the user's query.")
    success: bool = Field(default=True, description="Whether the task was successful.")


class FinishOutput(BaseModel):
    acknowledged: bool = True


def create_browser_tools(page: Page) -> list[BaseTool[BaseModel, BaseModel]]:
    """Create browser action tools bound to a Playwright page."""

    @create_fn_tool(
        name="navigate",
        description="Navigate to a URL. Use this to go to websites.",
    )
    async def navigate(url: str) -> NavigateOutput:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return NavigateOutput(
                success=True, current_url=page.url, title=await page.title()
            )
        except Exception as e:
            return NavigateOutput(
                success=False, current_url=page.url, title=f"Error: {e}"
            )

    @create_fn_tool(
        name="click",
        description="Click an element on the page using a CSS selector.",
    )
    async def click(selector: str) -> ClickOutput:
        try:
            await page.click(selector, timeout=5000)
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            return ClickOutput(success=True, message=f"Clicked {selector}")
        except Exception as e:
            return ClickOutput(success=False, message=f"Failed to click: {e}")

    @create_fn_tool(
        name="type_text",
        description="Type text into an input field using a CSS selector.",
    )
    async def type_text(selector: str, text: str) -> TypeTextOutput:
        try:
            await page.fill(selector, text, timeout=5000)
            return TypeTextOutput(success=True, message=f"Typed into {selector}")
        except Exception as e:
            return TypeTextOutput(success=False, message=f"Failed to type: {e}")

    @create_fn_tool(
        name="get_text",
        description="Extract text content from the page or a specific element.",
    )
    async def get_text(selector: str = "body") -> GetTextOutput:
        try:
            element = page.locator(selector)
            text = await element.inner_text(timeout=5000)
            # Truncate very long text
            if len(text) > 4000:
                text = text[:4000] + "... [truncated]"
            return GetTextOutput(text=text, success=True)
        except Exception as e:
            return GetTextOutput(text=f"Error: {e}", success=False)

    @create_fn_tool(
        name="get_page_info",
        description="Get current page URL and title.",
    )
    async def get_page_info() -> GetPageInfoOutput:
        return GetPageInfoOutput(url=page.url, title=await page.title(), success=True)

    @create_fn_tool(
        name="finish",
        description="Call this when the task is complete to return the final result.",
    )
    async def finish(result: str, success: bool = True) -> FinishOutput:  # noqa: ARG001
        return FinishOutput(acknowledged=True)

    return [navigate, click, type_text, get_text, get_page_info, finish]  # type: ignore


SYSTEM_PROMPT = """You are a web browsing agent. You can navigate websites, click elements, type text, and extract information.

Your goal is to accomplish the user's task by interacting with web pages.

Guidelines:
- Start by navigating to relevant websites
- Use get_text to read page content
- Use click and type_text to interact with forms and links
- When you have found the answer or completed the task, call 'finish' with the result
- Be efficient - don't navigate unnecessarily
- If you get stuck, try a different approach

Available tools:
- navigate(url): Go to a URL
- click(selector): Click an element (use CSS selectors like 'button', '#id', '.class')
- type_text(selector, text): Type into an input field
- get_text(selector): Extract text from page (default: full page)
- get_page_info(): Get current URL and title
- finish(result, success): Complete the task with final result

Always call 'finish' when done."""


class BrowsingTool(BaseTool[BrowsingToolInput, BrowsingToolOutput]):
    """
    A browsing agent tool that uses LLMClient to control a Playwright browser.

    This tool creates an agentic loop where the LLM decides which browser
    actions to take based on the current page state.
    """

    _name = "browsing_agent"
    description = "A browsing agent that can navigate websites and interact with web pages to accomplish tasks."
    _input = BrowsingToolInput
    _output = BrowsingToolOutput

    def __init__(
        self,
        llm_client: LLMClient,
        model: str | None = None,
        headless: bool = True,
    ) -> None:
        super().__init__()
        self.llm_client = llm_client
        self.model = model
        self.headless = headless

    def invoke(self, input: BrowsingToolInput) -> BrowsingToolOutput:
        """Sync execution - wraps async implementation."""

        return asyncio.run(self.ainvoke(input))

    async def ainvoke(self, input: BrowsingToolInput) -> BrowsingToolOutput:
        """Execute the browsing task asynchronously."""
        validated = self._validate_input(input)
        pages_visited: list[str] = []

        logger.info(
            "Starting browsing task | query={} | max_iterations={}",
            (
                validated.query[:50] + "..."
                if len(validated.query) > 50
                else validated.query
            ),
            validated.max_iterations,
        )

        async with async_playwright() as p:
            logger.debug("Launching browser | headless={}", self.headless)
            browser = await p.chromium.launch(headless=self.headless)
            page = await browser.new_page()

            try:
                # Navigate to start URL if provided
                if validated.start_url:
                    logger.info("Navigating to start URL: {}", validated.start_url)
                    await page.goto(validated.start_url, wait_until="domcontentloaded")
                    pages_visited.append(validated.start_url)

                # Create browser tools bound to this page
                browser_tools = create_browser_tools(page)

                # Run the agent loop
                result = await self._agent_loop(
                    query=validated.query,
                    tools=browser_tools,
                    max_iterations=validated.max_iterations,
                    pages_visited=pages_visited,
                )

                if result.success:
                    logger.success(
                        "Task completed | iterations={} | pages={}",
                        result.iterations_used,
                        len(result.pages_visited),
                    )
                else:
                    logger.warning(
                        "Task failed | iterations={} | result={}",
                        result.iterations_used,
                        result.result[:100],
                    )

                return result

            finally:
                logger.debug("Closing browser")
                await browser.close()

    async def _agent_loop(
        self,
        query: str,
        tools: list[BaseTool[BaseModel, BaseModel]],
        max_iterations: int,
        pages_visited: list[str],
    ) -> BrowsingToolOutput:
        """Run the agent loop until task completion or max iterations."""

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {query}"},
        ]

        tool_map = {tool.name: tool for tool in tools}

        for iteration in range(max_iterations):
            logger.debug(
                "Agent iteration {}/{} | messages={}",
                iteration + 1,
                max_iterations,
                len(messages),
            )

            # Get LLM decision
            response = await self.llm_client.agenerate(
                messages=messages,
                model=self.model,
                mode="tool_calling",
                tools=tools,
            )

            if not isinstance(response, ToolCallResponse) or not response.tool_calls:
                # No tool calls - shouldn't happen with tool_choice=required
                logger.error("No tool calls returned from LLM")
                return BrowsingToolOutput(
                    result="Agent stopped without calling finish",
                    success=False,
                    pages_visited=pages_visited,
                    iterations_used=iteration + 1,
                )

            # Process tool call
            tool_call = response.tool_calls[0]
            tool_name = tool_call.tool_name
            tool_args = tool_call.arguments

            logger.info(
                "Tool call | iteration={} | tool={} | args={}",
                iteration + 1,
                tool_name,
                str(tool_args)[:100],
            )

            # Check for finish (tool names are normalized to uppercase)
            if tool_name.upper() == "FINISH":
                logger.info("Agent called finish")
                return BrowsingToolOutput(
                    result=tool_args.get("result", "Task completed"),
                    success=tool_args.get("success", True),
                    pages_visited=pages_visited,
                    iterations_used=iteration + 1,
                )

            # Execute the tool
            tool = tool_map.get(tool_name)
            if tool is None:
                tool_result = f"Error: Unknown tool '{tool_name}'"
                logger.error("Unknown tool: {}", tool_name)
            else:
                try:
                    result = await tool.acall(tool_args)
                    tool_result = result.model_dump_json()
                    logger.debug("Tool result: {}", tool_result[:200])

                    # Track navigation (tool names are normalized to uppercase)
                    if tool_name.upper() == "NAVIGATE" and hasattr(
                        result, "current_url"
                    ):
                        url = getattr(result, "current_url")
                        pages_visited.append(url)
                        logger.info("Navigated to: {}", url)

                except Exception as e:
                    tool_result = f"Error executing {tool_name}: {e}"
                    logger.error("Tool execution error: {}", e)

            # Add assistant message with tool call
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": str(tool_args),
                            },
                        }
                    ],
                }
            )

            # Add tool result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

        # Max iterations reached
        logger.warning(
            "Max iterations reached | max={} | pages_visited={}",
            max_iterations,
            len(pages_visited),
        )
        return BrowsingToolOutput(
            result="Max iterations reached without completing the task",
            success=False,
            pages_visited=pages_visited,
            iterations_used=max_iterations,
        )

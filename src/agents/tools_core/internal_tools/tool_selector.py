"""
Tool selector that uses LLM to filter relevant tools based on user objective.

Takes a list of BaseTool and returns those relevant to accomplishing the objective.
Supports batching and parallel processing for large tool sets.
"""

import asyncio
import typing as t

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agents.configs import get_tools_core_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool
from agents.tools_core.llm_base_tool import LLMTool

# Load templates
_templates = get_tools_core_template_module("tool_selector.jinja")


class ToolInfo(BaseModel):
    """Information about a tool for selection purposes."""

    name: str = Field(description="The tool name")
    description: str = Field(description="The tool description")
    input_schema: dict[str, t.Any] | None = Field(
        default=None, description="JSON schema of the tool's input (optional)"
    )


class ToolSelectorInput(BaseModel):
    """Input for the tool selector."""

    objective: str = Field(description="The user objective to accomplish")
    tools: list[ToolInfo] = Field(description="List of available tools with their info")


class SelectedTool(BaseModel):
    """A tool selected by the LLM as relevant to the objective."""

    name: str = Field(description="Name of the selected tool")
    reason: str = Field(
        description="Brief explanation of why this tool is relevant to the objective"
    )


class ToolSelectorOutput(BaseModel):
    """Output from the tool selector."""

    selected_tools: list[SelectedTool] = Field(
        default_factory=list,
        description="List of tools selected as relevant to the objective. Empty if no tools are relevant.",
    )
    reasoning: str = Field(description="Overall reasoning for the tool selection")


class ToolSelector(LLMTool[ToolSelectorInput, ToolSelectorOutput]):
    """
    LLM-based tool that filters a list of tools based on user objective.

    Given a user objective and a list of available tools (as BaseTool instances),
    uses an LLM to determine which tools are relevant for accomplishing the objective.
    """

    _name = "tool_selector"
    description = "Selects relevant tools from a list based on user objective"
    _input = ToolSelectorInput
    _output = ToolSelectorOutput

    example_inputs: t.ClassVar[t.Sequence[ToolSelectorInput]] = (
        ToolSelectorInput(
            objective="Search for recent news about AI",
            tools=[
                ToolInfo(
                    name="web_search",
                    description="Search the web for information",
                    input_schema={"query": {"type": "string"}},
                ),
                ToolInfo(
                    name="calculator",
                    description="Perform mathematical calculations",
                    input_schema={"expression": {"type": "string"}},
                ),
            ],
        ),
    )

    example_outputs: t.ClassVar[t.Sequence[ToolSelectorOutput]] = (
        ToolSelectorOutput(
            selected_tools=[
                SelectedTool(
                    name="web_search",
                    reason="Required to search for recent AI news on the web",
                )
            ],
            reasoning="The objective requires searching for recent news, which the web_search tool can accomplish. The calculator tool is not relevant for this information retrieval task.",
        ),
    )

    def __init__(
        self,
        llm_client: LLMClient,
        model: str | None = None,
        batch_size: int = 100,
        parallel_mode: bool = False,
        include_input_schema: bool = True,
    ) -> None:
        """
        Initialize the ToolSelector.

        Args:
            llm_client: LLM client for generating responses
            model: Optional model override
            batch_size: Max tools per LLM call (default 100)
            parallel_mode: Process batches concurrently (default False)
            include_input_schema: Include tool input schemas in prompt (default True)
        """
        super().__init__(llm_client, model=model)
        self.batch_size = batch_size
        self.parallel_mode = parallel_mode
        self.include_input_schema = include_input_schema

    def format_messages(
        self, input: ToolSelectorInput
    ) -> list[ChatCompletionMessageParam]:
        """Format the prompt for tool selection."""
        prompt = _templates.tool_selector(
            objective=input.objective,
            tools=input.tools,
            output_schema=ToolSelectorOutput.model_json_schema(),
            include_input_schema=self.include_input_schema,
        )
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def tools_to_info(
        tools: t.Sequence[BaseTool[t.Any, t.Any]], include_input_schema: bool = True
    ) -> list[ToolInfo]:
        """Convert a sequence of BaseTool instances to ToolInfo list."""
        return [
            ToolInfo(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema() if include_input_schema else None,
            )
            for tool in tools
        ]

    def _batch_tools(self, tools: t.Sequence[ToolInfo]) -> list[list[ToolInfo]]:
        """Split tools into batches."""
        if len(tools) <= self.batch_size:
            return [list(tools)]
        return [
            list(tools[i : i + self.batch_size])
            for i in range(0, len(tools), self.batch_size)
        ]

    def _merge_results(
        self, results: t.Sequence[ToolSelectorOutput]
    ) -> ToolSelectorOutput:
        """Merge multiple batch results into a single output."""
        all_selected: list[SelectedTool] = []
        all_reasoning: list[str] = []

        for result in results:
            all_selected.extend(result.selected_tools)
            all_reasoning.append(result.reasoning)

        # Deduplicate by tool name
        seen_names: set[str] = set()
        unique_selected: list[SelectedTool] = []
        for tool in all_selected:
            if tool.name not in seen_names:
                seen_names.add(tool.name)
                unique_selected.append(tool)

        return ToolSelectorOutput(
            selected_tools=unique_selected,
            reasoning=" | ".join(all_reasoning)
            if len(all_reasoning) > 1
            else (all_reasoning[0] if all_reasoning else ""),
        )

    def select_from_tools(
        self,
        objective: str,
        tools: t.Sequence[BaseTool[t.Any, t.Any]],
    ) -> ToolSelectorOutput:
        """
        Select tools directly from BaseTool instances with batching support.

        If tools exceed batch_size, splits into batches and merges results.
        Note: Sync method does not support parallel_mode (use async for parallel).
        """
        tool_infos = self.tools_to_info(tools, self.include_input_schema)
        batches = self._batch_tools(tool_infos)

        if len(batches) == 1:
            return self.invoke(ToolSelectorInput(objective=objective, tools=batches[0]))

        # Process batches sequentially (sync doesn't support parallel)
        results = [
            self.invoke(ToolSelectorInput(objective=objective, tools=batch))
            for batch in batches
        ]
        return self._merge_results(results)

    async def aselect_from_tools(
        self,
        objective: str,
        tools: t.Sequence[BaseTool[t.Any, t.Any]],
    ) -> ToolSelectorOutput:
        """
        Async select tools with batching and optional parallel processing.

        If tools exceed batch_size, splits into batches.
        If parallel_mode=True, processes batches concurrently.
        """
        tool_infos = self.tools_to_info(tools, self.include_input_schema)
        batches = self._batch_tools(tool_infos)

        if len(batches) == 1:
            return await self.ainvoke(
                ToolSelectorInput(objective=objective, tools=batches[0])
            )

        if self.parallel_mode:
            # Process batches in parallel
            results = await asyncio.gather(
                *[
                    self.ainvoke(ToolSelectorInput(objective=objective, tools=batch))
                    for batch in batches
                ]
            )
        else:
            # Process batches sequentially
            results = [
                await self.ainvoke(ToolSelectorInput(objective=objective, tools=batch))
                for batch in batches
            ]

        return self._merge_results(results)

    def filter_tools(
        self,
        objective: str,
        tools: t.Sequence[BaseTool[t.Any, t.Any]],
    ) -> list[BaseTool[t.Any, t.Any]]:
        """
        Filter tools and return only the relevant BaseTool instances.
        """
        result = self.select_from_tools(objective, tools)
        selected_names = {t.name for t in result.selected_tools}
        return [tool for tool in tools if tool.name in selected_names]

    async def afilter_tools(
        self,
        objective: str,
        tools: t.Sequence[BaseTool[t.Any, t.Any]],
    ) -> list[BaseTool[t.Any, t.Any]]:
        """
        Async filter tools and return only the relevant BaseTool instances.
        """
        result = await self.aselect_from_tools(objective, tools)
        selected_names = {t.name for t in result.selected_tools}
        return [tool for tool in tools if tool.name in selected_names]

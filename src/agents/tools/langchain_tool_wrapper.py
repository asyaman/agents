import typing as t

from langchain_core.tools import BaseTool as LangchainBaseTool
from pydantic import BaseModel, Field

from agents.tools_core.base_tool import BaseTool


class DefaultOutputSchema(BaseModel):
    result: t.Any = Field(description="The result of running the tool")


# Placeholder input for class definition - will be overridden per instance
class _PlaceholderInput(BaseModel):
    pass


class LangchainToolWrapper(BaseTool[BaseModel, DefaultOutputSchema]):
    """Wraps a Langchain tool to work with the agents BaseTool interface."""

    _name = "langchain_tool"
    description = "Wrapped Langchain tool"
    _input: t.ClassVar[type[BaseModel]] = _PlaceholderInput
    _output: t.ClassVar[type[BaseModel]] = DefaultOutputSchema

    def __init__(
        self,
        langchain_tool: LangchainBaseTool,
        example_inputs: t.Sequence[BaseModel] = (),
        example_outputs: t.Sequence[DefaultOutputSchema] = (),
    ):
        self._langchain_tool = langchain_tool
        self.example_inputs = example_inputs
        self.example_outputs = example_outputs
        self.description = langchain_tool.description
        self._name = langchain_tool.name

        # Set _input dynamically from langchain tool's schema
        if langchain_tool.args_schema:
            self._input = langchain_tool.args_schema  # type: ignore

    def invoke(self, input: BaseModel) -> DefaultOutputSchema:
        """Synchronous invocation - runs the langchain tool."""
        result = self._langchain_tool.invoke(input.model_dump())  # type: ignore
        return DefaultOutputSchema(result=result)

    async def ainvoke(self, input: BaseModel) -> DefaultOutputSchema:
        """Asynchronous invocation - runs the langchain tool."""
        result = await self._langchain_tool.ainvoke(input.model_dump())  # type: ignore
        return DefaultOutputSchema(result=result)

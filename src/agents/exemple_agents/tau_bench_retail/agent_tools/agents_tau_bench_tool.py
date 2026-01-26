import typing as t

from pydantic import BaseModel, Field

from agents.example_agents.tau_bench_retail.tool_data.loader import load_data
from agents.example_agents.tau_bench_retail.tools.tool import Tool as TauBenchRetailTool
from agents.tools_core.base_tool import BaseTool
from agents.utilities.pydantic_utils import (
    create_example_from_schema,
    create_model_from_schema,
)


class DefaultOutput(BaseModel):
    result: t.Any = Field(description="The result of running the Tau Bench tool")


def create_tau_bench_tool(
    tool: TauBenchRetailTool,
    example_input: t.Sequence[BaseModel] | None = None,
    example_outputs: t.Sequence[DefaultOutput] | None = None,
) -> BaseTool[BaseModel, DefaultOutput]:
    """
    Factory function to create a BaseTool from a TauBenchRetailTool.

    Since the input schema is dynamic (determined at runtime from the tool),
    we create the tool class dynamically with the correct _input and _output.
    """
    tool_info = tool.get_info()["function"]
    tool_name = tool_info["name"]
    tool_description = tool_info["description"]

    # Build input model from tool's parameter schema
    input_model_schema = tool_info["parameters"]
    input_model_schema["title"] = (
        "".join(x.capitalize() for x in tool_name.lower().split("_")) + "Input"
    )
    input_model = create_model_from_schema(json_schema=input_model_schema)

    # Create example inputs/outputs
    if example_input:
        examples_in = tuple(example_input)
    else:
        examples_in = (
            input_model(**create_example_from_schema(json_schema=input_model_schema)),
        )

    if example_outputs:
        examples_out = tuple(example_outputs)
    else:
        examples_out = (
            DefaultOutput(
                **create_example_from_schema(DefaultOutput.model_json_schema())
            ),
        )

    # Define invoke method that captures the tool
    def _invoke(
        self: "BaseTool[BaseModel, DefaultOutput]", input: BaseModel
    ) -> DefaultOutput:
        validated = self._validate_input(input)
        result = tool.invoke(data=load_data(), **validated.model_dump())  # type: ignore
        return DefaultOutput(result=result)

    # Create the tool class dynamically
    TauBenchToolClass = type(
        f"TauBenchTool_{tool_name}",
        (BaseTool,),
        {
            "_name": tool_name,
            "description": tool_description,
            "_input": input_model,
            "_output": DefaultOutput,
            "example_inputs": examples_in,
            "example_outputs": examples_out,
            "invoke": _invoke,
        },
    )

    return TauBenchToolClass()  # type: ignore[return-value]

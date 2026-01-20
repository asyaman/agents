"""Tests for the simplified BaseTool implementation."""

import pytest
from pydantic import BaseModel

from agents.tools_core.base_tool import (
    BaseTool,
    InputValidationError,
    create_fn_tool,
    create_tool,
)


class SimpleInput(BaseModel):
    value: str


class SimpleOutput(BaseModel):
    result: str


class EchoTool(BaseTool[SimpleInput, SimpleOutput]):
    _name = "echo_tool"
    description = "Echoes the input value"
    _input = SimpleInput
    _output = SimpleOutput

    def invoke(self, input: SimpleInput) -> SimpleOutput:
        return SimpleOutput(result=input.value)


@pytest.fixture
def echo_tool() -> EchoTool:
    return EchoTool()


class TestBaseTool:
    def test_missing_input_raises_type_error(self):
        with pytest.raises(TypeError, match="must define '_input'"):

            class MissingInputTool(BaseTool[SimpleInput, SimpleOutput]):
                _name = "missing"
                description = "Missing input"
                _output = SimpleOutput

                def invoke(self, input: SimpleInput) -> SimpleOutput:
                    return SimpleOutput(result="")

    def test_missing_output_raises_type_error(self):
        with pytest.raises(TypeError, match="must define '_output'"):

            class MissingOutputTool(BaseTool[SimpleInput, SimpleOutput]):
                _name = "missing"
                description = "Missing output"
                _input = SimpleInput

                def invoke(self, input: SimpleInput) -> SimpleOutput:
                    return SimpleOutput(result="")

    def test_name_normalization(self, echo_tool: EchoTool):
        assert echo_tool.name == "ECHO_TOOL"
        assert echo_tool.raw_name == "echo_tool"

    def test_invoke(self, echo_tool: EchoTool):
        result = echo_tool.invoke(SimpleInput(value="hello"))
        assert result == SimpleOutput(result="hello")

    def test_call_with_model(self, echo_tool: EchoTool):
        result = echo_tool(SimpleInput(value="test"))
        assert result == SimpleOutput(result="test")

    def test_call_with_dict(self, echo_tool: EchoTool):
        result = echo_tool({"value": "from_dict"})
        assert result == SimpleOutput(result="from_dict")

    def test_invalid_input_raises_error(self, echo_tool: EchoTool):
        with pytest.raises(InputValidationError):
            echo_tool._validate_input("invalid")

    @pytest.mark.asyncio
    async def test_ainvoke(self, echo_tool: EchoTool):
        result = await echo_tool.ainvoke(SimpleInput(value="async_test"))
        assert result == SimpleOutput(result="async_test")

    def test_input_output_schema(self, echo_tool: EchoTool):
        input_schema = EchoTool.input_schema()
        assert "value" in input_schema["properties"]

        output_schema = EchoTool.output_schema()
        assert "result" in output_schema["properties"]

    def test_create_input(self, echo_tool: EchoTool):
        input_model = echo_tool.create_input(value="created")
        assert input_model.value == "created"


class TestCreateFnTool:
    def test_basic_function(self):
        @create_fn_tool(name="add", description="Adds two numbers")
        def add(x: int, y: int) -> int:
            return x + y

        result = add({"x": 1, "y": 2})
        assert result.root == 3  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_async_function(self):
        @create_fn_tool(name="async_add", description="Async add")
        async def async_add(x: int, y: int) -> int:
            return x + y

        result = await async_add.acall({"x": 3, "y": 4})
        assert result.root == 7  # type: ignore[attr-defined]


class TestCreateTool:
    def test_basic_function(self):
        @create_tool(name="echo", description="Echoes input")
        def echo(input: SimpleInput) -> SimpleOutput:
            return SimpleOutput(result=input.value)

        result = echo(SimpleInput(value="hello"))
        assert result == SimpleOutput(result="hello")

    def test_validates_single_parameter(self):
        with pytest.raises(ValueError, match="exactly one parameter"):

            @create_tool()
            def multi_param(a: SimpleInput, b: str) -> SimpleOutput:
                return SimpleOutput(result="")

    def test_validates_basemodel_types(self):
        with pytest.raises(TypeError, match="Input type must be a pydantic BaseModel"):

            @create_tool()
            def str_input(input: str) -> SimpleOutput:
                return SimpleOutput(result="")

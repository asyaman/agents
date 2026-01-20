"""Tests for the kwargs-based BaseTool implementation."""

import pytest
from pydantic import BaseModel

from agents.tools_core.base_tool_from_kwargs import (
    BaseTool,
    SignatureValidationError,
    simple_tool,
)


class ExplicitInput(BaseModel):
    x: int
    y: str = "default"


class ExplicitOutput(BaseModel):
    result: str


class EchoTool(BaseTool):
    _name = "echo_tool"
    description = "Echoes the value"

    def invoke(self, value: str) -> str:
        return value


class AddTool(BaseTool):
    _name = "add tool"
    description = "Adds two numbers"

    def invoke(self, x: int, y: int = 0) -> int:
        return x + y


class ExplicitModelTool(BaseTool):
    _name = "explicit_tool"
    description = "Tool with explicit models"
    _input = ExplicitInput
    _output = ExplicitOutput

    def invoke(self, x: int, y: str = "default") -> ExplicitOutput:
        return ExplicitOutput(result=f"{y}: {x}")


@pytest.fixture
def echo_tool() -> EchoTool:
    return EchoTool()


@pytest.fixture
def add_tool() -> AddTool:
    return AddTool()


@pytest.fixture
def explicit_tool() -> ExplicitModelTool:
    return ExplicitModelTool()


class TestBaseTool:
    def test_name_normalization(self, echo_tool: EchoTool, add_tool: AddTool):
        assert echo_tool.name == "ECHO_TOOL"
        assert echo_tool.raw_name == "echo_tool"
        assert add_tool.name == "ADD_TOOL"

    def test_invoke_with_kwargs(self, echo_tool: EchoTool):
        result = echo_tool.invoke(value="hello")
        assert result == "hello"

    def test_invoke_with_defaults(self, add_tool: AddTool):
        assert add_tool.invoke(x=5) == 5
        assert add_tool.invoke(x=5, y=3) == 8

    @pytest.mark.asyncio
    async def test_ainvoke(self, echo_tool: EchoTool):
        result = await echo_tool.ainvoke(value="async_test")
        assert result == "async_test"

    def test_input_model_from_signature(self, echo_tool: EchoTool):
        model = EchoTool.input_model()
        assert "value" in model.model_fields
        instance = model(value="test")
        assert instance.value == "test"

    def test_output_model_from_return_type(self, add_tool: AddTool):
        model = AddTool.output_model()
        instance = model(5)
        assert instance.root == 5

    def test_input_schema(self, add_tool: AddTool):
        schema = AddTool.input_schema()
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]

    def test_validate_input(self, add_tool: AddTool):
        validated = add_tool.validate_input(x=10, y=5)
        assert validated.x == 10
        assert validated.y == 5

    def test_validate_output(self, add_tool: AddTool):
        validated = add_tool.validate_output(42)
        assert validated.root == 42


class TestExplicitModels:
    def test_uses_explicit_input_model(self, explicit_tool: ExplicitModelTool):
        assert ExplicitModelTool.input_model() is ExplicitInput

    def test_uses_explicit_output_model(self, explicit_tool: ExplicitModelTool):
        assert ExplicitModelTool.output_model() is ExplicitOutput

    def test_invoke_returns_explicit_output(self, explicit_tool: ExplicitModelTool):
        result = explicit_tool.invoke(x=42, y="test")
        assert result == ExplicitOutput(result="test: 42")

    def test_signature_validation_error(self):
        with pytest.raises(SignatureValidationError, match="not found in method"):

            class BadTool(BaseTool):
                _name = "bad"
                description = "Bad tool"
                _input = ExplicitInput

                def invoke(self, z: int) -> str:  # x and y missing
                    return ""


class TestSimpleToolDecorator:
    def test_basic_sync_function(self):
        @simple_tool(name="multiply", description="Multiplies two numbers")
        def multiply(a: int, b: int) -> int:
            return a * b

        result = multiply.invoke(a=3, b=4)
        assert result == 12

    def test_infers_name_and_description(self):
        @simple_tool()
        def my_func(x: int) -> int:
            """Does something."""
            return x * 2

        assert my_func.name == "MY_FUNC"
        assert my_func.description == "Does something."

    @pytest.mark.asyncio
    async def test_async_function(self):
        @simple_tool(name="async_add", description="Async add")
        async def async_add(x: int, y: int) -> int:
            return x + y

        result = await async_add.ainvoke(x=3, y=4)
        assert result == 7

    def test_input_schema_from_decorator(self):
        @simple_tool(name="greet", description="Greets")
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        schema = greet.input_schema()
        assert "name" in schema["properties"]
        assert "greeting" in schema["properties"]

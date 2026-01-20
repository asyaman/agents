import numexpr  # type: ignore
from pydantic import BaseModel, Field

from agents.tools_core.base_tool import BaseTool


class CalculatorInput(BaseModel):
    numexpr: str = Field(
        description="""Input arg to the numexpr python library which is a mathematical expression consisting only on numeric values and the following mathematical operations:
- Bitwise operators (and, or, not, xor): &, |, ~, ^
- Comparison operators: <, <=, ==, !=, >=, >
- Unary arithmetic operators: -
- Binary arithmetic operators: +, -, *, /, **, %, <<, >>"""
    )


class CalculatorOutput(BaseModel):
    result: float = Field(description="The result of running the numexpr.")


class Calculator(BaseTool[CalculatorInput, CalculatorOutput]):
    _name = "calculator"
    description = """numexpr calculator from mathematical expressions consisting only on numeric values and bitwise, comparison, unary or binary operations"""
    _input = CalculatorInput
    _output = CalculatorOutput

    def invoke(self, input: CalculatorInput) -> CalculatorOutput:
        result: float = numexpr.evaluate(input.numexpr)  # type: ignore
        return CalculatorOutput(result=result)

    example_inputs = (
        CalculatorInput(numexpr="3+4"),
        CalculatorInput(numexpr="(4*5)/3"),
    )
    example_outputs = (CalculatorOutput(result=15.34),)

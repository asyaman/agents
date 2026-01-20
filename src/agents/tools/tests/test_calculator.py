"""
Tests for agents/tools/calculator.py

Tests:
- test_calculator_name_and_description: Name and description
- test_calculator_simple_addition: Basic arithmetic
- test_calculator_complex_expression: Complex expressions
- test_calculator_input_output_schemas: Schema structure
"""

from agents.tools.calculator import Calculator, CalculatorInput, CalculatorOutput


class TestCalculator:
    def test_name_and_description(self):
        calc = Calculator()
        assert calc.name == "CALCULATOR"
        assert "calculator" in calc.description.lower()

    def test_simple_addition(self):
        calc = Calculator()
        result = calc.invoke(CalculatorInput(numexpr="3 + 4"))
        assert result.result == 7.0

    def test_complex_expression(self):
        calc = Calculator()
        result = calc.invoke(CalculatorInput(numexpr="(10 * 5) / 2"))
        assert result.result == 25.0

    def test_power_operation(self):
        calc = Calculator()
        result = calc.invoke(CalculatorInput(numexpr="2 ** 8"))
        assert result.result == 256.0

    def test_input_output_schemas(self):
        calc = Calculator()
        input_schema = calc.input_schema()
        assert "numexpr" in input_schema["properties"]

        output_schema = calc.output_schema()
        assert "result" in output_schema["properties"]

    def test_example_inputs_outputs(self):
        calc = Calculator()
        assert len(calc.example_inputs) > 0
        assert len(calc.example_outputs) > 0
        assert isinstance(calc.example_inputs[0], CalculatorInput)
        assert isinstance(calc.example_outputs[0], CalculatorOutput)

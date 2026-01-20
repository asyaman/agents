"""
Simple tool base class with single pydantic model input/output pattern.

This is a simplified alternative that:
- Uses a single pydantic model for invoke/ainvoke input (not **kwargs)
- Requires explicit _input and _output model definitions
- Derives schemas directly from the pydantic models
- Validates input type at runtime
"""

import asyncio
import inspect
import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, RootModel, ValidationError, create_model
from pydantic.errors import PydanticSchemaGenerationError
from pydantic.functional_serializers import PlainSerializer

from agents.utilities.utils import normalize_tool_name

# Type variables for input/output models
InputT = t.TypeVar("InputT", bound=BaseModel)
OutputT = t.TypeVar("OutputT", bound=BaseModel)


class InputValidationError(Exception):
    """Raised when input validation fails."""

    pass


class BaseTool(ABC, t.Generic[InputT, OutputT]):
    """
    A simple tool base class with single pydantic model input/output.

    Subclasses must:
    1. Set `_name` and `description` as class attributes
    2. Define `_input` and `_output` as pydantic BaseModel types
    3. Override `invoke()` for sync implementation or `ainvoke()` for async

    Example:
        class MyInput(BaseModel):
            x: int
            y: str = "default"

        class MyOutput(BaseModel):
            result: str

        class MyTool(BaseTool[MyInput, MyOutput]):
            _name = "my_tool"
            description = "Does something useful"
            _input = MyInput
            _output = MyOutput

            def invoke(self, input: MyInput) -> MyOutput:
                return MyOutput(result=f"{input.y}: {input.x}")
    """

    _name: str
    description: str

    # Required: explicit pydantic models for input/output
    _input: t.ClassVar[type[BaseModel]]
    _output: t.ClassVar[type[BaseModel]]

    # Examples for documentation
    example_inputs: t.ClassVar[t.Sequence[BaseModel]] = ()
    example_outputs: t.ClassVar[t.Sequence[BaseModel]] = ()

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        """Validate the subclass configuration on definition."""
        super().__init_subclass__(**kwargs)

        # Skip validation for abstract classes
        if inspect.isabstract(cls):
            return

        # Validate that _input and _output are defined
        if not hasattr(cls, "_input") or cls._input is None:
            raise TypeError(
                f"{cls.__name__} must define '_input' as a pydantic BaseModel type"
            )
        if not hasattr(cls, "_output") or cls._output is None:
            raise TypeError(
                f"{cls.__name__} must define '_output' as a pydantic BaseModel type"
            )

        # Validate that they are BaseModel subclasses
        if not (inspect.isclass(cls._input) and issubclass(cls._input, BaseModel)):
            raise TypeError(
                f"{cls.__name__}._input must be a pydantic BaseModel subclass, "
                f"got {cls._input}"
            )
        if not (inspect.isclass(cls._output) and issubclass(cls._output, BaseModel)):
            raise TypeError(
                f"{cls.__name__}._output must be a pydantic BaseModel subclass, "
                f"got {cls._output}"
            )

    @property
    def name(self) -> str:
        """Normalized tool name for LLM compatibility."""
        return normalize_tool_name(self._name)

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def raw_name(self) -> str:
        """Original tool name without normalization."""
        return self._name

    def _validate_input(self, input: t.Any) -> InputT:
        """Validate that input is the correct type."""
        expected_type = self._input

        # If already the correct type, return as-is
        if isinstance(input, expected_type):
            return t.cast(InputT, input)

        # If it's a dict, try to construct the model
        if isinstance(input, dict):
            try:
                return t.cast(InputT, expected_type(**input))
            except ValidationError as e:
                raise InputValidationError(
                    f"Failed to validate input dict as {expected_type.__name__}: {e}"
                ) from e

        # If it's another BaseModel, try to convert
        if isinstance(input, BaseModel):
            try:
                return t.cast(InputT, expected_type(**input.model_dump()))
            except ValidationError as e:
                raise InputValidationError(
                    f"Failed to convert {type(input).__name__} to {expected_type.__name__}: {e}"
                ) from e

        raise InputValidationError(
            f"Expected {expected_type.__name__} or dict, got {type(input).__name__}"
        )

    @abstractmethod
    def invoke(self, input: InputT) -> OutputT:
        """
        Synchronous execution of the tool.

        Args:
            input: Pydantic model matching self._input type

        Returns:
            Pydantic model matching self._output type
        """
        ...

    async def ainvoke(self, input: InputT) -> OutputT:
        """
        Asynchronous execution of the tool.

        Default implementation calls invoke() in a thread pool.
        Override for native async implementation.

        Args:
            input: Pydantic model matching self._input type

        Returns:
            Pydantic model matching self._output type
        """
        validated_input = self._validate_input(input)
        return await asyncio.to_thread(self.invoke, validated_input)

    @classmethod
    def input_model(cls) -> type[InputT]:
        """Get the input pydantic model type."""
        return t.cast(type[InputT], cls._input)

    @classmethod
    def output_model(cls) -> type[OutputT]:
        """Get the output pydantic model type."""
        return t.cast(type[OutputT], cls._output)

    @classmethod
    def input_schema(cls) -> dict[str, t.Any]:
        """Get the JSON schema for the input model."""
        return cls._input.model_json_schema()

    @classmethod
    def output_schema(cls) -> dict[str, t.Any]:
        """Get the JSON schema for the output model."""
        return cls._output.model_json_schema()

    def create_input(self, **kwargs: t.Any) -> InputT:
        """Convenience method to create an input model from kwargs."""
        return t.cast(InputT, self._input(**kwargs))

    def __call__(self, input: InputT | dict[str, t.Any]) -> OutputT:
        """
        Call the tool synchronously.
        Accepts either the input model or a dict that will be converted.
        """
        validated_input = self._validate_input(input)
        return self.invoke(validated_input)

    async def acall(self, input: InputT | dict[str, t.Any]) -> OutputT:
        """
        Call the tool asynchronously.
        Accepts either the input model or a dict that will be converted.
        """
        validated_input = self._validate_input(input)
        return await self.ainvoke(validated_input)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, description={self.description!r})"


def _build_model_from_signature(
    fn: t.Callable[..., t.Any],
    schema_name: str,
    exclude_params: tuple[str, ...] = ("self", "cls", "kwargs"),
) -> type[BaseModel]:
    """Build a pydantic model from a function's parameter signature."""
    sig = inspect.signature(fn)

    fields: dict[str, t.Any] = {}
    for name, param in sig.parameters.items():
        # Skip self, cls, **kwargs, etc.
        if name in exclude_params or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        # Get type annotation or default to Any
        annotation = (
            param.annotation if param.annotation != inspect.Parameter.empty else t.Any
        )

        # Handle default values
        if param.default != inspect.Parameter.empty:
            fields[name] = (annotation, param.default)
        else:
            fields[name] = (annotation, ...)

    try:
        return create_model(schema_name, **fields)
    except PydanticSchemaGenerationError as e:
        raise ValueError(
            f"Cannot create pydantic model from signature of {fn.__name__}. "
            "Use supported primitive types or provide explicit _input model."
        ) from e


def _build_output_model_from_return(
    fn: t.Callable[..., t.Any],
    schema_name: str,
) -> type[BaseModel]:
    """Build a pydantic model from a function's return type annotation."""
    sig = inspect.signature(fn)
    return_annotation = sig.return_annotation

    # Handle missing return annotation
    if return_annotation == inspect.Signature.empty:
        return_annotation = t.Any

    # If already a BaseModel subclass, return it directly
    try:
        if inspect.isclass(return_annotation) and issubclass(
            t.cast(type, return_annotation), BaseModel
        ):
            return t.cast(type[BaseModel], return_annotation)
    except TypeError:
        pass  # Some types like generics can't be used with issubclass

    # Try to create a RootModel for primitive/generic types
    try:
        model = create_model(
            schema_name,
            __base__=RootModel[return_annotation],  # type: ignore[valid-type]
        )
        return model
    except PydanticSchemaGenerationError:
        # Fallback for unsupported types
        model = create_model(
            schema_name,
            result=(t.Annotated[t.Any, PlainSerializer(str)], ...),
        )
        return model


# Convenience decorator for creating tools from functions with **kwargs signature
def create_fn_tool(
    name: str | None = None,
    description: str | None = None,
    example_inputs: t.Sequence[dict[str, t.Any]] | None = None,
    example_outputs: t.Sequence[t.Any] | None = None,
) -> t.Callable[[t.Callable[..., t.Any]], BaseTool[BaseModel, BaseModel]]:
    """
    Decorator to create a BaseTool from a function with **kwargs signature.

    The input model is auto-generated from the function's parameter signature.
    The output model is auto-generated from the function's return type annotation.

    Example:
        @create_fn_tool(name="add_numbers", description="Adds two numbers")
        def add(x: int, y: int) -> int:
            return x + y

        # Usage:
        result = add_tool({"x": 1, "y": 2})  # Returns the int result
    """

    def decorator(fn: t.Callable[..., t.Any]) -> BaseTool[BaseModel, BaseModel]:
        tool_name = name or fn.__name__
        tool_description = description or fn.__doc__ or ""

        # Build input/output models from function signature
        input_model = _build_model_from_signature(fn, schema_name=f"{tool_name}Input")
        output_model = _build_output_model_from_return(
            fn, schema_name=f"{tool_name}Output"
        )

        is_async = inspect.iscoroutinefunction(fn)

        def _invoke_sync(
            self: BaseTool[BaseModel, BaseModel], input: BaseModel
        ) -> BaseModel:
            validated = self._validate_input(input)
            result = fn(**validated.model_dump())
            # Wrap result in output model if needed
            if isinstance(result, BaseModel):
                return result
            # For RootModel, wrap the result
            return output_model(result)  # type: ignore[call-arg]

        def _invoke_async_placeholder(
            self: BaseTool[BaseModel, BaseModel], input: BaseModel
        ) -> BaseModel:
            raise NotImplementedError(
                f"{tool_name} is async-only. Use ainvoke() or acall()."
            )

        async def _ainvoke_async(
            self: BaseTool[BaseModel, BaseModel], input: BaseModel
        ) -> BaseModel:
            validated = self._validate_input(input)
            result = await fn(**validated.model_dump())
            # Wrap result in output model if needed
            if isinstance(result, BaseModel):
                return result
            # For RootModel, wrap the result
            return output_model(result)  # type: ignore[call-arg]

        # Build class attributes
        class_attrs: dict[str, t.Any] = {
            "_name": tool_name,
            "_input": input_model,
            "_output": output_model,
            "description": tool_description,
            "example_inputs": example_inputs or (),
            "example_outputs": example_outputs or (),
        }

        if is_async:
            class_attrs["invoke"] = _invoke_async_placeholder
            class_attrs["ainvoke"] = _ainvoke_async
        else:
            class_attrs["invoke"] = _invoke_sync

        # Create class using type()
        FunctionTool = type(
            "FunctionTool",
            (BaseTool,),
            class_attrs,
        )

        return FunctionTool()  # type: ignore[return-value]

    return decorator


# Convenience decorator for creating tools from functions with pydantic input and ouputs
def create_tool(
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
    output_model: type[BaseModel] | None = None,
    example_inputs: t.Sequence[BaseModel] | None = None,
    example_outputs: t.Sequence[BaseModel] | None = None,
) -> t.Callable[[t.Callable[[BaseModel], BaseModel]], BaseTool[BaseModel, BaseModel]]:
    """
    Decorator to create a BaseTool from a function.

    The function must have a single pydantic model parameter and return a pydantic model.
    If input_model/output_model are not provided, they are inferred from type hints.

    Example:
        class AddInput(BaseModel):
            x: int
            y: int

        class AddOutput(BaseModel):
            result: int

        @create_tool(name="add_numbers", description="Adds two numbers")
        def add(input: AddInput) -> AddOutput:
            return AddOutput(result=input.x + input.y)
    """

    def decorator(
        fn: t.Callable[[BaseModel], BaseModel],
    ) -> BaseTool[BaseModel, BaseModel]:
        tool_name = name or fn.__name__
        tool_description = description or fn.__doc__ or ""

        # Infer input/output models from function signature if not provided
        hints = t.get_type_hints(fn)
        params = list(inspect.signature(fn).parameters.values())

        # Get input model
        inferred_input = input_model
        if inferred_input is None:
            if len(params) != 1:
                raise ValueError(
                    f"Function {fn.__name__} must have exactly one parameter, "
                    f"got {len(params)}"
                )
            param_name = params[0].name
            if param_name in hints:
                inferred_input = hints[param_name]
            else:
                raise ValueError(
                    f"Function {fn.__name__} parameter '{param_name}' must have a type hint"
                )

        # Get output model
        inferred_output = output_model
        if inferred_output is None:
            if "return" not in hints:
                raise ValueError(f"Function {fn.__name__} must have a return type hint")
            inferred_output = hints["return"]

        # Validate they are BaseModel subclasses
        if not (
            inspect.isclass(inferred_input) and issubclass(inferred_input, BaseModel)
        ):
            raise TypeError(
                f"Input type must be a pydantic BaseModel subclass, got {inferred_input}"
            )
        if not (
            inspect.isclass(inferred_output) and issubclass(inferred_output, BaseModel)
        ):
            raise TypeError(
                f"Output type must be a pydantic BaseModel subclass, got {inferred_output}"
            )

        # Create the tool class dynamically
        is_async = inspect.iscoroutinefunction(fn)

        def _invoke_sync(
            self: BaseTool[BaseModel, BaseModel], input: BaseModel
        ) -> BaseModel:
            validated = self._validate_input(input)
            return fn(validated)

        def _invoke_async_placeholder(
            self: BaseTool[BaseModel, BaseModel], input: BaseModel
        ) -> BaseModel:
            raise NotImplementedError(
                f"{tool_name} is async-only. Use ainvoke() or acall()."
            )

        async def _ainvoke_async(
            self: BaseTool[BaseModel, BaseModel], input: BaseModel
        ) -> BaseModel:
            validated = self._validate_input(input)
            return await fn(validated)  # type: ignore[misc]

        # Build class attributes
        class_attrs: dict[str, t.Any] = {
            "_name": tool_name,
            "_input": inferred_input,
            "_output": inferred_output,
            "description": tool_description,
            "example_inputs": example_inputs or (),
            "example_outputs": example_outputs or (),
        }

        if is_async:
            class_attrs["invoke"] = _invoke_async_placeholder
            class_attrs["ainvoke"] = _ainvoke_async
        else:
            class_attrs["invoke"] = _invoke_sync

        # Create class using type()
        FunctionTool = type(
            "FunctionTool",
            (BaseTool,),
            class_attrs,
        )

        return FunctionTool()  # type: ignore[return-value]

    return decorator

"""
Simple tool base class with straightforward **kwargs invoke pattern.

This is a simplified alternative to base_tool.py that:
- Uses **kwargs for invoke/ainvoke instead of a single pydantic model
- Auto-generates input/output models from method signatures
- Optionally validates against explicit _input/_output class attributes
"""

import asyncio
import inspect
import typing as t
from abc import ABC

from pydantic import (
    BaseModel,
    PydanticSchemaGenerationError,
    RootModel,
    create_model,
)
from pydantic.functional_serializers import PlainSerializer

from agents.utilities.utils import normalize_tool_name


class SignatureValidationError(Exception):
    """Raised when method signature doesn't match declared model."""

    pass


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


def _validate_signature_against_model(
    fn: t.Callable[..., t.Any],
    model: type[BaseModel],
    exclude_params: tuple[str, ...] = ("self", "cls", "kwargs"),
) -> None:
    """Validate that function signature matches the provided pydantic model."""
    sig = inspect.signature(fn)
    model_fields = set(model.model_fields.keys())

    sig_params: dict[str, t.Any] = {}
    for name, param in sig.parameters.items():
        if name in exclude_params or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        sig_params[name] = (
            param.annotation if param.annotation != inspect.Parameter.empty else t.Any
        )

    sig_param_names = set(sig_params.keys())

    # Check for missing parameters
    missing_in_sig = model_fields - sig_param_names
    if missing_in_sig:
        raise SignatureValidationError(
            f"Model defines fields {missing_in_sig} not found in method signature"
        )

    # Check for extra parameters (warning only - they might have defaults)
    extra_in_sig = sig_param_names - model_fields
    if extra_in_sig:
        # Only error if they don't have defaults
        sig_obj = inspect.signature(fn)
        for param_name in extra_in_sig:
            param = sig_obj.parameters[param_name]
            if param.default == inspect.Parameter.empty:
                raise SignatureValidationError(
                    f"Method has required parameter '{param_name}' not defined in model"
                )


class BaseTool(ABC):
    """
    A simple tool base class with **kwargs invoke pattern.

    Subclasses should:
    1. Set `_name` and `description` as class attributes
    2. Override `invoke()` for sync implementation or `ainvoke()` for async
    3. Optionally define `_input` and `_output` as BaseModel types for validation
    4. Optionally provide `example_inputs` and `example_outputs`

    Example:
        class MyTool(BaseTool):
            _name = "my_tool"
            description = "Does something useful"

            def invoke(self, x: int, y: str = "default") -> str:
                return f"{y}: {x}"

    With explicit models:
        class MyInput(BaseModel):
            x: int
            y: str = "default"

        class MyTool(BaseTool):
            _name = "my_tool"
            description = "Does something useful"
            _input = MyInput

            def invoke(self, x: int, y: str = "default") -> str:
                return f"{y}: {x}"
    """

    _name: str
    description: str

    # Optional: explicit pydantic models for input/output
    # If provided, the invoke signature will be validated against them
    _input: t.ClassVar[type[BaseModel] | None] = None
    _output: t.ClassVar[type[BaseModel] | None] = None

    # Examples for documentation
    example_inputs: t.ClassVar[t.Sequence[dict[str, t.Any]]] = ()
    example_outputs: t.ClassVar[t.Sequence[t.Any]] = ()

    # Cache for generated models
    _cached_input_model: t.ClassVar[type[BaseModel] | None] = None
    _cached_output_model: t.ClassVar[type[BaseModel] | None] = None

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        """Validate the subclass configuration on definition."""
        super().__init_subclass__(**kwargs)

        # Skip validation for abstract classes
        if inspect.isabstract(cls):
            return

        # Validate _input model against invoke signature if provided
        if cls._input is not None and hasattr(cls, "invoke"):
            invoke_method = cls.invoke
            # Only validate if invoke is overridden (not the base class version)
            if invoke_method is not BaseTool.invoke:
                _validate_signature_against_model(invoke_method, cls._input)

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

    def invoke(self, **kwargs: t.Any) -> t.Any:
        """
        Synchronous execution of the tool.

        Override this method for sync implementations.
        If only ainvoke is overridden, this will raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sync invoke(). "
            "Override invoke() for sync execution or use ainvoke() for async."
        )

    async def ainvoke(self, **kwargs: t.Any) -> t.Any:
        """
        Asynchronous execution of the tool.

        Default implementation calls invoke() in a thread pool.
        Override for native async implementation.
        """
        return await asyncio.to_thread(self.invoke, **kwargs)

    @classmethod
    def input_model(cls) -> type[BaseModel]:
        """
        Get the pydantic model representing the input schema.

        If `_input` is defined, returns it directly.
        Otherwise, generates a model from the invoke() signature.
        """
        # Return explicit model if provided
        if cls._input is not None:
            return cls._input

        # Use cached model if available
        if cls._cached_input_model is not None:
            return cls._cached_input_model

        # Generate from invoke signature
        invoke_method = cls.invoke
        if invoke_method is BaseTool.invoke:
            raise ValueError(
                f"{cls.__name__} must override invoke() or define _input model"
            )

        model = _build_model_from_signature(
            invoke_method, schema_name=f"{cls.__name__}Input"
        )
        cls._cached_input_model = model
        return model

    @classmethod
    def output_model(cls) -> type[BaseModel]:
        """
        Get the pydantic model representing the output schema.

        If `_output` is defined, returns it directly.
        Otherwise, generates a model from the invoke() return type.
        """
        # Return explicit model if provided
        if cls._output is not None:
            return cls._output

        # Use cached model if available
        if cls._cached_output_model is not None:
            return cls._cached_output_model

        # Generate from invoke return type
        invoke_method = cls.invoke
        if invoke_method is BaseTool.invoke:
            raise ValueError(
                f"{cls.__name__} must override invoke() or define _output model"
            )

        model = _build_output_model_from_return(
            invoke_method, schema_name=f"{cls.__name__}Output"
        )
        cls._cached_output_model = model
        return model

    @classmethod
    def input_schema(cls) -> dict[str, t.Any]:
        """Get the JSON schema for the input model."""
        return cls.input_model().model_json_schema()

    @classmethod
    def output_schema(cls) -> dict[str, t.Any]:
        """Get the JSON schema for the output model."""
        return cls.output_model().model_json_schema()

    def validate_input(self, **kwargs: t.Any) -> BaseModel:
        """Validate input kwargs against the input model."""
        return self.input_model()(**kwargs)

    def validate_output(self, result: t.Any) -> BaseModel:
        """Validate output against the output model."""
        output_model = self.output_model()
        if issubclass(output_model, RootModel):
            return output_model(root=result)
        elif isinstance(result, BaseModel):
            return result
        else:
            # Assume single-field model with 'result' field
            return output_model(result=result)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, description={self.description!r})"


# Convenience decorator for creating tools from functions
def simple_tool(
    name: str | None = None,
    description: str | None = None,
    example_inputs: t.Sequence[dict[str, t.Any]] | None = None,
    example_outputs: t.Sequence[t.Any] | None = None,
) -> t.Callable[[t.Callable[..., t.Any]], BaseTool]:
    """
    Decorator to create a BaseTool from a function.

    Example:
        @simple_tool(name="add_numbers", description="Adds two numbers")
        def add(x: int, y: int) -> int:
            return x + y
    """

    def decorator(fn: t.Callable[..., t.Any]) -> BaseTool:
        tool_name = name or fn.__name__
        tool_description = description or fn.__doc__ or ""

        class FunctionTool(BaseTool):
            _name = tool_name

        FunctionTool.description = tool_description
        FunctionTool.example_inputs = example_inputs or ()
        FunctionTool.example_outputs = example_outputs or ()

        # Determine if function is async
        if inspect.iscoroutinefunction(fn):

            async def ainvoke_impl(self: BaseTool, **kwargs: t.Any) -> t.Any:
                return await fn(**kwargs)

            FunctionTool.ainvoke = ainvoke_impl

            # Generate models from the async function
            FunctionTool._cached_input_model = _build_model_from_signature(
                fn, schema_name=f"{tool_name}Input"
            )
            FunctionTool._cached_output_model = _build_output_model_from_return(
                fn, schema_name=f"{tool_name}Output"
            )
        else:

            def invoke_impl(self: BaseTool, **kwargs: t.Any) -> t.Any:
                return fn(**kwargs)

            FunctionTool.invoke = invoke_impl

            # Generate models from the sync function
            FunctionTool._cached_input_model = _build_model_from_signature(
                fn, schema_name=f"{tool_name}Input"
            )
            FunctionTool._cached_output_model = _build_output_model_from_return(
                fn, schema_name=f"{tool_name}Output"
            )

        return FunctionTool()

    return decorator

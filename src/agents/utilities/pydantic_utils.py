"""This is used to create Pydantic models based on json schemas"""

import random
import typing as t
from types import MappingProxyType

from polyfactory.factories.pydantic_factory import ModelFactory
from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
)

RESERVED_NAMES = ["validate"]

DefaultHandling = t.Literal[
    "optional_default", "no_type_conflicts", "no_default", "add_default_as_type"
]


def create_model_from_schema(
    json_schema: dict[str, t.Any],
    add_examples: bool = False,
    default_handling: DefaultHandling | None = None,
    definitions: dict[str, t.Any] | None = None,
) -> t.Type[BaseModel]:
    """Create a Pydantic model from a JSON schema.

    This function takes a JSON schema as input and dynamically creates a Pydantic model class
    based on the schema. It supports various JSON schema features such as nested objects,
    referenced definitions, custom configurations, custom base classes, custom validators, and more.

    Args:
        json_schema: A dictionary representing the JSON schema.
        add_examples: Flag indicating whether examples for fields shoudl be added
        definitions: Top level reference definitions if the json schema is
            of format "$defs" and "$refs".
            We need to pass through the nested structure the top level reference
            definitions to access the nested schemas.
    Returns:
        A dynamically created Pydantic model class based on the provided JSON schema.

    """
    model_name = json_schema.get("title", "DynamicModel")
    required_fields: list[str] = json_schema.get("required") or []
    definitions = definitions or json_schema.get("$defs") or {}

    field_definitions: dict[str, t.Any] = {}
    for name, prop in (json_schema.get("properties") or {}).items():  # type: ignore
        type, pydantic_field_kwargs = json_schema_to_pydantic_field(
            name=name,  # type: ignore
            json_schema=prop,  # type: ignore
            required=required_fields,
            definitions=definitions,
            add_examples=add_examples,
            default_handling=default_handling,
        )
        field_definitions[name] = (type, Field(**pydantic_field_kwargs))
    return create_model(
        model_name,
        **field_definitions,
    )


def is_instance_of_type(obj: t.Any, expected_type: t.Any) -> bool:
    """Works with generics so better than isinstance."""
    try:
        TypeAdapter(expected_type).validate_python(obj)
        return True
    except ValidationError:
        return False


def get_default(
    json_schema: dict[str, t.Any],
    name: str,
    type_: t.Any,
    required: list[str],
    default_handling: DefaultHandling | None,
) -> tuple[t.Any, bool, t.Any]:
    """
    Determine the default value for a field based on the schema and default handling.

    Handling types:
    `no_type_conflicts`: the default kept only when there is not conflict of types
    `no_default`: all default values are discarded, meaning those fields are flagged as required.
    `optional_default`: all default values are discarded, meaning those fields are flagged as required, additionally the tyoe is set Optional[type].
    `add_default_as_type`: If the default type is not the same as the field type it adds it as to a union, and field being required is unchanged.
    `None`: Does nothing.
    """
    set_optional = name not in required

    if "default" not in json_schema:
        return ..., set_optional, type_

    default = json_schema["default"]

    if default_handling is None:
        return default, set_optional, type_

    if default_handling == "add_default_as_type" and not is_instance_of_type(
        default, type_
    ):
        return default, set_optional, t.Union[type_, type(default)]
    if default_handling == "no_type_conflicts" and not is_instance_of_type(
        default, type_
    ):
        return ..., False, type_
    if default_handling == "no_default":
        return ..., False, type_
    if default_handling == "optional_default":
        return ..., True, type_

    return default, set_optional, type_


def json_schema_to_pydantic_field(
    name: str,
    json_schema: t.Dict[str, t.Any],
    required: t.List[str],
    definitions: t.Dict[str, t.Any] | None,
    add_examples: bool,
    default_handling: DefaultHandling | None,
) -> t.Tuple[t.Any, dict[str, t.Any]]:
    description = json_schema.get("description")
    examples = json_schema.get("examples", [])

    if name in RESERVED_NAMES:
        name = f"{name}_"
        alias = name
    else:
        alias = None

    type_: t.Any = parse_type_to_pydantic_type(
        json_schema=json_schema,
        definitions=definitions,
        add_examples=add_examples,
        default_handling=default_handling,
    )

    default, set_optional, type_ = get_default(
        json_schema, name, type_, required, default_handling
    )
    if set_optional:
        type_ = t.Optional[type_]

    field_kwargs: dict[str, t.Any] = {
        "description": description,
        "default": default,
        "alias": alias,
    }

    if add_examples:
        field_kwargs["examples"] = examples

    if "$ref" in json_schema:
        ref_name: str = json_schema.get("$ref", "").replace("#/$defs/", "")
        if definitions:
            ref_schema: dict[str, t.Any] = definitions.get(ref_name) or {}
        else:
            raise ValueError(
                f'A reference schema should be provided for {ref_name} in "$defs"'
            )

        def_type, def_pydantic_field = json_schema_to_pydantic_field(
            name=name,
            json_schema=ref_schema,
            required=required,
            definitions=definitions,
            add_examples=add_examples,
            default_handling=default_handling,
        )
        def_pydantic_field.update(field_kwargs)
        return def_type, def_pydantic_field

    return (type_, field_kwargs)


def parse_type_to_pydantic_type(
    json_schema: dict[str, t.Any],
    *,
    definitions: t.Dict[str, t.Any] | None,
    add_examples: bool,
    default_handling: DefaultHandling | None,
    name: t.Optional[str] = None,
) -> t.Any:
    type_ = json_schema.get("type")
    if type_ == "string":
        return str
    elif type_ == "integer":
        return int
    elif type_ == "number":
        return float
    elif type_ == "boolean":
        return bool
    elif type_ == "array":
        items_schema = json_schema.get("items")
        if items_schema:
            item_type = parse_type_to_pydantic_type(
                items_schema,
                name=name,
                definitions=definitions,
                add_examples=add_examples,
                default_handling=default_handling,
            )
            return list[item_type]  # type: ignore[valid-type]
        else:
            return list
    elif type_ == "object":
        properties = json_schema.get("properties")
        if properties:
            json_schema_ = json_schema.copy()
            if json_schema_.get("title") is None:
                json_schema_["title"] = name
            nested_model = create_model_from_schema(
                json_schema_,
                definitions=definitions,
                add_examples=add_examples,
                default_handling=default_handling,
            )
            return nested_model
        else:
            return dict
    elif type_ == "null":
        return t.Optional[t.Any]
    elif type_ is None and "oneOf" in json_schema:
        return t.Union[
            *(
                parse_type_to_pydantic_type(
                    option,
                    definitions=definitions,
                    add_examples=add_examples,
                    default_handling=default_handling,
                )
                for option in json_schema.get("oneOf", [])
            )
        ]

    elif type_ is None and "anyOf" in json_schema:
        options = json_schema.get("anyOf", [])
        if {"type": "null"} in options:
            return t.Optional[
                t.Union[
                    *(
                        parse_type_to_pydantic_type(
                            sub_schema,
                            definitions=definitions,
                            add_examples=add_examples,
                            default_handling=default_handling,
                        )
                        for sub_schema in options
                        if sub_schema != {"type": "null"}  # noqa: F821
                    )
                ]
            ]
        return t.Union[
            *(
                parse_type_to_pydantic_type(
                    sub_schema,
                    definitions=definitions,
                    add_examples=add_examples,
                    default_handling=default_handling,
                )
                for sub_schema in options
            )
        ]
    elif type_ is None:
        return t.Any
    else:
        raise ValueError(f"Unsupported JSON schema type: {type_} from {json_schema}")


def create_example_from_schema(
    json_schema: dict[str, t.Any],
) -> dict[str, t.Any]:
    """Create an example from Pydantic model.
    Returns:
        A dynamically created example for provided pydacntic model.

    """
    return {
        name: convert_type_to_sample(json_schema=prop, name=name)
        for name, prop in json_schema.get("properties", {}).items()
    }


def convert_type_to_sample(
    json_schema: dict[str, t.Any],
    name: t.Optional[str] = None,
) -> t.Any:
    """
    Generates a sample value based on a JSON schema definition.

    This function analyzes the provided JSON schema and returns a representative sample value for the specified type.
    It supports common JSON schema types such as `string`, `integer`, `number`, `boolean`, `array`, `object`, `null`,
    and a fallback for undefined types.
    """
    type_ = json_schema.get("type")

    if type_ == "string":
        return "..."
    elif type_ == "integer":
        return random.randrange(10)
    elif type_ == "number":
        return random.uniform(0.5, 1.5)
    elif type_ == "boolean":
        return bool(random.randint(0, 1))
    elif type_ == "array":
        items_schema = json_schema.get("items")
        if items_schema:
            item_type = convert_type_to_sample(items_schema, name=name)
            return [item_type]
        else:
            return list
    elif type_ == "object":
        properties = json_schema.get("properties")
        if properties:
            json_schema_ = json_schema.copy()
            if json_schema_.get("title") is None:
                json_schema_["title"] = name
            nested_model = create_example_from_schema(json_schema_)
            return nested_model
        else:
            return {}
    elif type_ == "null":
        return None
    elif type_ is None and "oneOf" in json_schema:
        first_option = json_schema["oneOf"][0]
        return convert_type_to_sample(first_option)
    elif type_ is None and "anyOf" in json_schema:
        first_option = json_schema["anyOf"][0]
        return convert_type_to_sample(first_option)
    elif type_ is None:
        return "..."
    else:
        raise ValueError(f"Unsupported JSON schema type: {type_} from {json_schema}")


# The default examples for types.
# These should be llm friendly and are designed to be used in prompts.
defaults_provider_map: MappingProxyType[type, t.Callable[[], t.Any]] = MappingProxyType(
    {
        str: lambda: "...",
        int: lambda: random.randrange(0, 10),
        bool: lambda: random.choice([True, False]),
        float: lambda: random.uniform(0.5, 1.5),
        dict: lambda: {},
        list: lambda: [],
    }
)


def create_example_from_model_factory[T](cls: type[T]) -> type[ModelFactory[T]]:
    """Creates a factory for generating examples with suitable defaults for a Pydantic model."""

    class ModelDefaultExamplesFactory(ModelFactory[cls]):
        @classmethod
        def get_provider_map(cls) -> dict[t.Any, t.Callable[[], t.Any]]:
            # Overrides the old providers map with the new ones.
            providers = super().get_provider_map()
            return {**providers, **defaults_provider_map}

    ModelDefaultExamplesFactory.__name__ = f"{cls.__name__}DefaultExamplesFactory"
    return ModelDefaultExamplesFactory


def create_example_from_model[T](cls: type[T]) -> T:
    """Creates an example with suitable defaults for a Pydantic model.
    Note this function is more inefficient since it creates new factories every time.
    Prefer using `create_example_from_model_factory` and `build` for multiple examples.
    """
    factory = create_example_from_model_factory(cls)()
    return factory.build()


def process_schema(
    schema: dict[str, t.Any],
    /,
    hide_defaults: bool = False,
    make_fields_required: bool = False,
    enforce_additional_properties: bool = False,
) -> dict[str, t.Any]:
    """
    General processor for schema transformations. Applies optional flags to:
    - Hide default values from the schema (and updates the description to say what the default is).
    - Ensure additionalProperties is set to False if not explicitly defined.
    - Make all fields required.
    Args:
        schema (dict[str, t.Any]): The schema to process.
        hide_defaults (bool): Whether to hide default values from the schema.
        enforce_additional_properties (bool): Whether to enforce additionalProperties as False.

    Returns:
        dict[str, t.Any]: The processed schema.

    See docs at the bottom of this file for more examples.
    """

    def _hide_default(field: dict[str, t.Any], field_name: str):
        """Process individual fields based on flags."""
        # _make_required should run first to correctly create the msg to add to the description.
        if "default" in field:
            default = field["default"]

            # If the field is not required, it can be omitted from the schema output.
            # Thus we add "...if unset"
            if field_name not in schema.get("required", []):
                msg = f"Defaults to '{default}' if unset."
            # If the field is required, still mention what the default case is in the description to reduce value hallucinations.
            else:
                msg = f"Default to {default}."

            if "description" in field:
                field["description"] += " | " + msg
            else:
                field["description"] = msg

            del field["default"]

    def _enforce_additional_properties(field: dict[str, t.Any]):
        """Enforce additionalProperties as False if not explicitly defined."""
        if "properties" in field and "additionalProperties" not in field:
            field["additionalProperties"] = False

    def _make_required(field_name: str):
        """Add field to required list if make_fields_required is enabled."""
        if "required" not in schema:
            schema["required"] = []
        if field_name not in schema["required"]:
            schema["required"].append(
                field_name
            )  # pyright: ignore[reportUnknownMemberType]

    if enforce_additional_properties:
        _enforce_additional_properties(schema)

    if "properties" in schema:
        for field_name, field in schema["properties"].items():
            # Make required should be ran before hide default
            if make_fields_required:
                _make_required(field_name)

            if hide_defaults:
                _hide_default(field, field_name)

            # Recursively apply
            if "properties" in field:
                process_schema(
                    field,
                    hide_defaults,
                    make_fields_required,
                    enforce_additional_properties,
                )
            elif "items" in field and isinstance(field["items"], dict):
                process_schema(
                    field["items"],
                    hide_defaults,
                    make_fields_required,
                    enforce_additional_properties,
                )

    if "$defs" in schema:
        for definition in schema["$defs"].values():
            process_schema(
                definition,
                hide_defaults,
                make_fields_required,
                enforce_additional_properties,
            )

    return schema


######################################################################
############## Docs for the schema processing ##############
######################################################################

# Example class default schema (compare to following the cases)
"""A.model_json_schema()
{
    "properties": {
        "a": {
            "title": "A",
            "type": "string"
        },
        "b": {
            "default": 1,
            "description": "Description.",
            "title": "B",
            "type": "integer"
        }
    },
    "required": [
        "a"
    ],
    "title": "A",
    "type": "object"
}
"""

# Remove schema defaults
"""process_schema(A.model_json_schema(), hide_defaults=True)
{
    "properties": {
        "a": {
            "title": "A",
            "type": "string"
        },
        "b": {
            "description": "Description. Defaults to '1' if unset.",
            "title": "B",
            "type": "integer"
        }
    },
    "required": [
        "a"
    ],
    "title": "A",
    "type": "object"
}
"""

# Make Fields Required
"""process_schema(A.model_json_schema(), make_fields_required=True)
{
    "properties": {
        "a": {
            "title": "A",
            "type": "string"
        },
        "b": {
            "default": 1,
            "description": "Description.",
            "title": "B",
            "type": "integer"
        }
    },
    "required": [
        "a",
        "b"
    ],
    "title": "A",
    "type": "object"
}
"""

# Enforce additionalProperties
"""process_schema(A.model_json_schema(), enforce_additional_properties=True)
{
    "properties": {
        "a": {
            "title": "A",
            "type": "string"
        },
        "b": {
            "default": 1,
            "description": "Description.",
            "title": "B",
            "type": "integer"
        }
    },
    "required": [
        "a"
    ],
    "title": "A",
    "type": "object",
    "additionalProperties": false
}
"""

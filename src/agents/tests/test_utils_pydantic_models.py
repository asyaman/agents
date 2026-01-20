"""
Tests for agents/utilities/pydantic_utils.py

Tests:
- test_parse_type_to_pydantic_type: Convert JSON schema types to Python types
- test_json_schema_to_pydantic_field_default_handling: Handle defaults in schema conversion
- test_create_model_schema_different_nested_model_representations: Nested model schemas
- test_default_and_required_combination: Default values and required fields interaction
- test_convert_type_to_sample: Generate sample values from schema types
- test_create_example_from_model_factory: Create examples using polyfactory
- test_create_example_from_model: Create examples from pydantic models
- test_create_example_from_model_union: Handle union types in examples
- test_user_defined_preprocessor_is_applied: Custom schema preprocessors
- test_make_required_with_all_optional_fields: Make optional fields required
"""

import random
import typing as t
from unittest.mock import Mock, patch

import pytest
from faker import Faker
from polyfactory.factories.pydantic_factory import ModelFactory
from pydantic import BaseModel, ConfigDict, Field

from agents.utilities.pydantic_utils import (
    DefaultHandling,
    convert_type_to_sample,
    create_example_from_model,
    create_example_from_model_factory,
    create_model_from_schema,
    get_default,
    parse_type_to_pydantic_type,
    process_schema,
)


@pytest.mark.parametrize(
    ["schema", "expected"],
    [
        ({"type": "string"}, str),
        ({"type": "integer"}, int),
        ({"type": "boolean"}, bool),
        ({"type": "number"}, float),
        ({"type": "array"}, list),
        ({"type": "null"}, t.Optional[t.Any]),
        ({"type": None}, t.Any),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, t.Optional[str]),
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, t.Union[str, int]),
        (
            {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]},
            t.Optional[t.Union[str, int]],
        ),
        (
            {
                "type": "array",
                "items": {"type": "string"},
            },
            list[str],
        ),
    ],
)
def test_parse_type_to_pydantic_type(schema: dict[str, t.Any], expected: t.Any):
    result = parse_type_to_pydantic_type(
        schema, definitions=None, add_examples=False, default_handling=None
    )
    assert result == expected


OPTIONAL = True
REQUIRED = False


# name, schema, type, required, default_handling, expected
@pytest.mark.parametrize(
    ["name", "schema", "type", "required", "default_handling", "expected"],
    [
        # Default handling: standard
        # Name in required
        ("first_name", {"default": 10}, int, ["first_name"], None, (10, False, int)),
        ("first_name", {}, int, ["first_name"], None, (Ellipsis, False, int)),
        # Default handling: standard
        # Name not in required
        ("surname", {}, int, ["first_name"], None, (Ellipsis, True, int)),
        ("surname", {"default": 10}, int, ["first_name"], None, (10, True, int)),
        # Default handling: "no_type_conflicts"
        # Name not in required
        (
            "surname",
            {"default": "a"},
            int,
            ["first_name"],
            "no_type_conflicts",
            (Ellipsis, REQUIRED, int),
        ),
        # Name in required
        (
            "first_name",
            {"default": "a"},
            int,
            ["first_name"],
            "no_type_conflicts",
            (Ellipsis, REQUIRED, int),
        ),
        # Default handling: "no_default"
        # Name not in required
        (
            "surname",
            {"default": 10},
            int,
            ["first_name"],
            "no_default",
            (Ellipsis, REQUIRED, int),
        ),
        (
            "surname",
            {"default": "a"},
            int,
            ["first_name"],
            "no_default",
            (Ellipsis, REQUIRED, int),
        ),
        # Default handling: "optional_default"
        # Name not in required
        (
            "surname",
            {"default": 10},
            int,
            ["first_name"],
            "optional_default",
            (Ellipsis, OPTIONAL, int),
        ),
        (
            "surname",
            {"default": None},
            int,
            ["first_name"],
            "optional_default",
            (Ellipsis, OPTIONAL, int),
        ),
        (
            "surname",
            {"default": None},
            str,
            ["first_name"],
            "add_default_as_type",
            (None, OPTIONAL, str | None),
        ),
        (
            "surname",
            {"default": None},
            str,
            ["surname"],
            "add_default_as_type",
            (None, REQUIRED, str | None),
        ),
        # Try with generics
        (
            "surname",
            {"default": [1]},
            list[str] | None,
            ["surname"],
            "add_default_as_type",
            ([1], REQUIRED, list[str] | None | list),
        ),
    ],
)
def test_json_schema_to_pydantic_field_default_handling(
    name: str,
    schema: dict[str, t.Any],
    type: t.Any,
    required: list[str],
    default_handling: DefaultHandling,
    expected: t.Any,
):
    result = get_default(
        name=name,
        json_schema=schema,
        required=required,
        type_=type,
        default_handling=default_handling,
    )
    assert result == expected


json_schema_with_def = {
    "$defs": {
        "Age": {
            "properties": {
                "current_age": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "description": "Age of the person",
                    "title": "Current Age",
                },
                "birthyear": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": 2000,
                    "description": "Birth year of the person",
                    "title": "Birthyear",
                },
            },
            "required": ["current_age"],
            "title": "Age",
            "type": "object",
        },
        "Person": {
            "properties": {
                "person_name": {
                    "description": "name of the Person",
                    "title": "Person Name",
                    "type": "string",
                },
                "age": {"$ref": "#/$defs/Age", "description": "age of the Person"},
            },
            "required": ["person_name", "age"],
            "title": "Person",
            "type": "object",
        },
    },
    "properties": {
        "number": {
            "default": 3,
            "description": "House number of the person",
            "title": "Number",
            "type": "integer",
        },
        "person": {
            "$ref": "#/$defs/Person",
            "description": "Person residing at number",
        },
    },
    "required": ["person"],
    "title": "House",
    "type": "object",
}

json_schema = {
    "properties": {
        "number": {
            "default": 3,
            "description": "House number of the person",
            "title": "Number",
            "type": "integer",
        },
        "person": {
            "properties": {
                "person_name": {
                    "description": "name of the Person",
                    "title": "Person Name",
                    "type": "string",
                },
                "age": {
                    "properties": {
                        "current_age": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "description": "Age of the person",
                            "title": "Current Age",
                        },
                        "birthyear": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": 2000,
                            "description": "Birth year of the person",
                            "title": "Birthyear",
                        },
                    },
                    "required": ["current_age"],
                    "title": "Age",
                    "type": "object",
                    "description": "age of the Person",
                },
            },
            "required": ["person_name", "age"],
            "title": "Person",
            "type": "object",
            "description": "Person residing at number",
        },
    },
    "required": ["person"],
    "title": "House",
    "type": "object",
}


def test_create_model_schema_different_nested_model_representations():
    assert (
        create_model_from_schema(json_schema).model_json_schema()  # type: ignore
        == create_model_from_schema(json_schema_with_def).model_json_schema()  # type: ignore
        == {
            "$defs": {
                "Age": {
                    "properties": {
                        "birthyear": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": 2000,
                            "description": "Birth year of the person",
                            "title": "Birthyear",
                        },
                        "current_age": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "description": "Age of the person",
                            "title": "Current Age",
                        },
                    },
                    "required": ["current_age"],
                    "title": "Age",
                    "type": "object",
                },
                "Person": {
                    "properties": {
                        "age": {
                            "$ref": "#/$defs/Age",
                            "description": "age of the Person",
                        },
                        "person_name": {
                            "description": "name of the Person",
                            "title": "Person Name",
                            "type": "string",
                        },
                    },
                    "required": ["person_name", "age"],
                    "title": "Person",
                    "type": "object",
                },
            },
            "properties": {
                "number": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": 3,
                    "description": "House number of the person",
                    "title": "Number",
                },
                "person": {
                    "$ref": "#/$defs/Person",
                    "description": "Person residing at number",
                },
            },
            "required": ["person"],
            "title": "House",
            "type": "object",
        }
    )

    assert (
        create_model_from_schema(
            json_schema,  # type: ignore
            default_handling="no_default",
        ).model_json_schema()
        == create_model_from_schema(
            json_schema_with_def,  # type: ignore
            default_handling="no_default",
        ).model_json_schema()
        == {
            "$defs": {
                "Age": {
                    "properties": {
                        "birthyear": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Birth year of the person",
                            "title": "Birthyear",
                        },
                        "current_age": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "description": "Age of the person",
                            "title": "Current Age",
                        },
                    },
                    "required": ["current_age", "birthyear"],
                    "title": "Age",
                    "type": "object",
                },
                "Person": {
                    "properties": {
                        "age": {
                            "$ref": "#/$defs/Age",
                            "description": "age of the Person",
                        },
                        "person_name": {
                            "description": "name of the Person",
                            "title": "Person Name",
                            "type": "string",
                        },
                    },
                    "required": ["person_name", "age"],
                    "title": "Person",
                    "type": "object",
                },
            },
            "properties": {
                "number": {
                    "description": "House number of the person",
                    "title": "Number",
                    "type": "integer",
                },
                "person": {
                    "$ref": "#/$defs/Person",
                    "description": "Person residing at number",
                },
            },
            "required": ["number", "person"],
            "title": "House",
            "type": "object",
        }
    )


@pytest.mark.parametrize(
    ["val", "cond"],
    [
        (
            {"default": 10},
            lambda field_schema, required: field_schema.get("default") == 10  # type: ignore
            and "field_name" not in required,  # type: ignore
        ),
        (
            {"default": Ellipsis},
            lambda field_schema, required: "default" not in field_schema  # type: ignore
            and "field_name" in required,
        ),
        (
            {"default": None},
            lambda field_schema, required: field_schema.get("default") is None  # type: ignore
            and "field_name" not in required,
        ),
        (
            {},
            lambda field_schema, required: "default" not in field_schema  # type: ignore
            and "field_name" in required,
        ),
    ],
)
def test_default_and_required_combination(val: dict[str, t.Any], cond: t.Any):
    with patch(
        "agents.utilities.pydantic_utils.json_schema_to_pydantic_field",
        Mock(return_value=(int, val)),
    ):
        result_model = create_model_from_schema(
            json_schema={"properties": {"field_name": {}}},
            add_examples=False,
            definitions=None,
        )
        result_model_schema = result_model.model_json_schema().get("properties")
        required = result_model.model_json_schema().get("required", [])
        field_schema = result_model_schema.get("field_name")  # type: ignore
        assert cond(field_schema, required)


@pytest.mark.parametrize(
    ["schema", "expected", "mock_patch"],
    [
        ({"type": "string"}, "...", None),
        ({"type": "integer"}, 5, patch("random.randrange", return_value=5)),
        ({"type": "number"}, 1.0, patch("random.uniform", return_value=1.0)),
        ({"type": "boolean"}, True, patch("random.randint", return_value=1)),
        ({"type": "boolean"}, False, patch("random.randint", return_value=0)),
        ({"type": "array", "items": {"type": "string"}}, ["..."], None),
        ({"type": "array"}, list, None),
        ({"type": "object"}, {}, None),
        ({"type": "null"}, None, None),
        ({"anyOf": [{"type": "string"}]}, "...", None),
    ],
)
def test_convert_type_to_sample(
    schema: dict[str, t.Any], expected: t.Any, mock_patch: t.Any
):
    if mock_patch:
        with mock_patch:
            result = convert_type_to_sample(schema)
    else:
        result = convert_type_to_sample(schema)
    assert result == expected


class SubObj(BaseModel):
    x: int
    y: str


class Obj(BaseModel):
    a: int
    b: str
    c: float
    d: tuple[int]
    e: dict[str, int]
    f: dict[str, dict[str, int]]
    g: list[int]
    h: dict  # type: ignore
    i: list  # type: ignore
    j: list[dict]  # type: ignore
    sub: SubObj


@pytest.fixture(autouse=True)
def polyfactory_random_seeds(faker: Faker):
    """Sets up random seeds for polyfactory"""
    random.seed(0)
    ModelFactory.__faker__ = faker
    ModelFactory.__random__.seed(0)


def test_create_example_from_model_factory():
    factory = create_example_from_model_factory(Obj)()
    obj = factory.build()
    expected = Obj.model_construct(
        a=6,
        b="...",
        c=pytest.approx(1.2579544029403),  # type: ignore[reportAttributeAccessIssue]
        d=(6,),
        e={"...": 0},
        f={"...": {"...": 4}},
        g=[8],
        h={},
        i=[],
        j=[{}],
        sub=SubObj(x=7, y="..."),
    )
    assert obj == expected


def test_create_example_from_model():
    obj = create_example_from_model(SubObj)
    expected = SubObj.model_construct(x=6, y="...")
    assert obj == expected


class A(BaseModel):
    a: int


class B(BaseModel):
    b: int


class UnionObj(BaseModel):
    obj: t.Union[A, B]


def test_create_example_from_model_union():
    obj = create_example_from_model(UnionObj)
    expected = UnionObj(obj=B(b=6))
    assert obj == expected


def custom_schema_fn(schema: dict[t.Any, t.Any]):
    schema["custom_schema_level_key"] = "custom_schema_level_value"


class SchemaWithPreprocessor(BaseModel):
    a: int = Field(
        default=1,
        description="A description",
        json_schema_extra={"custom_field_key": "custom_field_value"},
    )


class PreprocessorNested(BaseModel):
    model_config = ConfigDict(json_schema_extra=custom_schema_fn)

    a: SchemaWithPreprocessor


def test_user_defined_preprocessor_is_applied():
    result = process_schema(
        PreprocessorNested.model_json_schema(),
        hide_defaults=False,
        make_fields_required=False,
        enforce_additional_properties=False,
    )
    assert result == {
        "$defs": {
            "SchemaWithPreprocessor": {
                "properties": {
                    "a": {
                        "custom_field_key": "custom_field_value",
                        "default": 1,
                        "description": "A description",
                        "title": "A",
                        "type": "integer",
                    }
                },
                "title": "SchemaWithPreprocessor",
                "type": "object",
            }
        },
        "custom_schema_level_key": "custom_schema_level_value",
        "properties": {"a": {"$ref": "#/$defs/SchemaWithPreprocessor"}},
        "required": ["a"],
        "title": "PreprocessorNested",
        "type": "object",
    }


class ModelOptionals(BaseModel):
    x: str | None = None
    y: int | str = 1


def test_make_required_with_all_optional_fields():
    result = process_schema(
        ModelOptionals.model_json_schema(),
        hide_defaults=False,
        make_fields_required=True,
        enforce_additional_properties=False,
    )
    assert result == {
        "properties": {
            "x": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "X",
            },
            "y": {
                "anyOf": [{"type": "integer"}, {"type": "string"}],
                "default": 1,
                "title": "Y",
            },
        },
        "title": "ModelOptionals",
        "type": "object",
        "required": ["x", "y"],
    }


# More tests for `process_schema` is in `test_llms.py`, which tests that the different llm modes are calling
# `process_schema` correctly and outputting correct schemas.

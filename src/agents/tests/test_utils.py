"""
Tests for agents/utils.py

Tests:
- test_extract_json_subtexts: Extract JSON from markdown code blocks
- test_normalize_tool_name: Normalize tool names to uppercase with underscores
"""

from agents.utilities.utils import extract_json_subtexts, normalize_tool_name


def test_extract_json_subtexts():
    assert (
        extract_json_subtexts("preceding text ```json THIS IS TEXT``` trailing")
        == "THIS IS TEXT"
    )
    assert (
        extract_json_subtexts("preceding text ```json PARTIAL TEXT") == "PARTIAL TEXT"
    )
    assert extract_json_subtexts("no json here") == "no json here"


def test_normalize_tool_name():
    assert normalize_tool_name("my_tool") == "MY_TOOL"
    assert normalize_tool_name("my tool") == "MY_TOOL"
    assert normalize_tool_name("My Tool Name") == "MY_TOOL_NAME"

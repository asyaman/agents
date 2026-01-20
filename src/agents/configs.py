"""Centralized configuration for the agents package."""

from pathlib import Path

import jinja2
from jinja2.sandbox import SandboxedEnvironment


AGENTS_ROOT = Path(__file__).parent
LLM_TOOLS_PROMPTS_DIR = AGENTS_ROOT / "tools" / "llm_tools" / "prompts"
TOOLS_CORE_PROMPTS_DIR = AGENTS_ROOT / "tools_core" / "internal_tools" / "prompts"
AGENT_TOOL_PROMPTS_DIR = AGENTS_ROOT / "agent_tool" / "prompts"


def _create_jinja_env(prompts_dir: Path) -> SandboxedEnvironment:
    """Create a sandboxed jinja environment for a prompts directory.

    Uses sandboxed environment to prevent arbitrary code execution in templates.
    StrictUndefined raises errors on undefined variables instead of silent empty strings.
    See: https://jinja.palletsprojects.com/en/3.1.x/sandbox/
    """
    return SandboxedEnvironment(
        loader=jinja2.FileSystemLoader(prompts_dir),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


# Jinja environments for different prompt directories
llm_tools_jinja_env = _create_jinja_env(LLM_TOOLS_PROMPTS_DIR)
tools_core_jinja_env = _create_jinja_env(TOOLS_CORE_PROMPTS_DIR)
agent_tool_jinja_env = _create_jinja_env(AGENT_TOOL_PROMPTS_DIR)


def get_template(name: str) -> jinja2.Template:
    """Load a jinja template by name from the llm_tools prompts directory."""
    return llm_tools_jinja_env.get_template(name)


def get_template_module(name: str):
    """Load a jinja template module (for macro access) from llm_tools prompts."""
    return llm_tools_jinja_env.get_template(name).module


def get_tools_core_template(name: str) -> jinja2.Template:
    """Load a jinja template by name from the tools_core prompts directory."""
    return tools_core_jinja_env.get_template(name)


def get_tools_core_template_module(name: str):
    """Load a jinja template module (for macro access) from tools_core prompts."""
    return tools_core_jinja_env.get_template(name).module


def get_agent_tool_template(name: str) -> jinja2.Template:
    """Load a jinja template by name from the agent_tool prompts directory."""
    return agent_tool_jinja_env.get_template(name)


def get_agent_tool_template_module(name: str):
    """Load a jinja template module (for macro access) from agent_tool prompts."""
    return agent_tool_jinja_env.get_template(name).module

"""
LLM wrappers that format tool output as natural language.

FormatToolOutput: Formats raw tool output as natural language (simple, no chunking)
    - Input: Tool's output type (e.g., SearchOutput)
    - Output: NLOutput (success, result, error)
    - Base: LLMTool
    - Use case: SearchOutput(results=[...]) → "Found 3 tutorials about..."

SimplifyToolOutput: Token-aware output simplification with chunking
    - Input: Tool's output type (e.g., SearchOutput)
    - Output: NLOutput | raw output (conditional based on size)
    - Base: LLMTool
    - Features: Token counting, pass-through for small outputs, chunking for large
    - Use case: Large tool outputs that need smart simplification
"""

import asyncio
import typing as t

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from agents.configs import get_tools_core_template_module
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool
from agents.tools_core.llm_base_tool import LLMTool
from agents.tools_core.internal_tools.nl_models import LLMError, NLOutput

# Type variables for tool input/output
ToolInputT = t.TypeVar("ToolInputT", bound=BaseModel)
ToolOutputT = t.TypeVar("ToolOutputT", bound=BaseModel)

# Load templates
_templates = get_tools_core_template_module("tool_output_formatter.jinja")

# Default token thresholds
_DEFAULT_TOKEN_LOWER_BOUND = 5000  # Below this: pass through raw output
_DEFAULT_TOKEN_UPPER_BOUND = 90000  # Above this: reject as too large
_DEFAULT_CHUNK_SIZE = 30000  # Token size per chunk
_DEFAULT_CHUNK_OVERLAP = 1000  # Overlap between chunks


class OutputTooLargeError(Exception):
    """Raised when tool output exceeds the maximum allowed token size."""

    def __init__(self, tool_name: str, token_count: int, max_tokens: int):
        self.tool_name = tool_name
        self.token_count = token_count
        self.max_tokens = max_tokens
        super().__init__(
            f"Tool '{tool_name}' output of {token_count} tokens exceeds max {max_tokens}"
        )


class ChunkSimplificationError(Exception):
    """Raised when chunk simplification fails."""

    def __init__(self, message: str, error: LLMError):
        self.error = error
        super().__init__(message)


class FormatToolOutput(LLMTool[ToolOutputT, NLOutput]):
    """
    Wraps a tool's output with LLM that formats it as natural language.

    Takes the raw tool output and task objective, uses LLM to format a response.
    """

    _input = BaseModel  # Placeholder, overwritten in __init__ with tool's output type
    _output = NLOutput

    def __init__(
        self,
        tool: BaseTool[ToolInputT, ToolOutputT],
        task: str,
        llm_client: LLMClient,
        model: str | None = None,
    ) -> None:
        super().__init__(llm_client, model=model)

        self.tool = tool
        self.task = task
        self._tool_output_type = tool._output
        self._input = tool._output  # Input is the tool's output type

        self._name = f"llm_format_{tool.name}"
        self.description = f"LLM formats the output of the tool: {tool.description}"

        self.example_inputs = tool.example_outputs

        # Build example outputs for the prompt
        self._example_outputs = [
            NLOutput(success=True, result="...").model_dump_json(),
            NLOutput(
                success=False,
                error=LLMError(
                    error="...",
                    type_of_error="Incomplete tool output",
                    content="The tool output provides ... .",
                    suggested_fix="...",
                ),
            ).model_dump_json(),
        ]

    def format_messages(self, input: ToolOutputT) -> list[ChatCompletionMessageParam]:
        prompt = _templates.tool_output_wrapper(
            task=self.task,
            tool_name=self.tool.name,
            tool_description=self.tool.description,
            output_schema=self.tool._output.model_json_schema(),
            tool_output=input.model_dump_json(),
            response_schema=NLOutput.model_json_schema(),
            error_schema=LLMError.model_json_schema(),
            examples=self._example_outputs,
        )
        return [{"role": "user", "content": prompt}]


class SimplifyToolOutput(LLMTool[ToolOutputT, NLOutput]):
    """
    Token-aware tool output simplifier with chunking support.

    Features:
    - Pass-through: Returns raw output if below token_lower_bound (saves LLM calls)
    - Simplification: Uses LLM to simplify outputs between lower and upper bounds
    - Chunking: Splits large outputs and processes chunks in parallel
    - Hierarchical: Simplifies chunks, then combines and simplifies again
    - Protection: Raises error if output exceeds token_upper_bound

    Args:
        tool: The tool whose output to simplify
        task: The objective/task description for context
        llm_client: LLM client for simplification
        model: Optional model override (uses tiktoken for token counting)
        token_lower_bound: Skip simplification below this (default 5000)
        token_upper_bound: Reject outputs above this (default 90000)
        chunk_size: Token size per chunk (default 30000)
        chunk_overlap: Overlap between chunks (default 1000)
        parallel_chunks: Process chunks in parallel (default True)
    """

    _input = BaseModel  # Placeholder, overwritten in __init__
    _output = NLOutput

    def __init__(
        self,
        tool: BaseTool[ToolInputT, ToolOutputT],
        task: str,
        llm_client: LLMClient,
        model: str | None = None,
        token_lower_bound: int = _DEFAULT_TOKEN_LOWER_BOUND,
        token_upper_bound: int = _DEFAULT_TOKEN_UPPER_BOUND,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        parallel_chunks: bool = True,
    ) -> None:
        super().__init__(llm_client, model=model)

        self.tool = tool
        self.task = task
        self._tool_output_type = tool._output
        self._input = tool._output

        self._name = f"llm_simplify_{tool.name}"
        self.description = f"LLM simplifies output of tool: {tool.description}"

        # Token thresholds
        self.token_lower_bound = token_lower_bound
        self.token_upper_bound = token_upper_bound
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parallel_chunks = parallel_chunks

        # Determine model name for tiktoken
        self._tiktoken_model = model or llm_client._default_model or "gpt-4"

        self.example_inputs = tool.example_outputs
        self._example_outputs = [
            NLOutput(success=True, result="...").model_dump_json(),
            NLOutput(
                success=False,
                error=LLMError(
                    error="...",
                    type_of_error="Incomplete tool output",
                    content="The tool output provides ... .",
                    suggested_fix="...",
                ),
            ).model_dump_json(),
        ]

    def _get_tokenizer(self) -> tiktoken.Encoding:
        """Get tiktoken encoder for the model."""
        try:
            return tiktoken.encoding_for_model(self._tiktoken_model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            return tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._get_tokenizer().encode(text))

    def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get text splitter configured for the model."""
        try:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name=self._tiktoken_model,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        except KeyError:
            # Fallback for unknown models
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def format_messages(self, input: ToolOutputT) -> list[ChatCompletionMessageParam]:
        """Format messages for simplification."""
        prompt = _templates.tool_output_wrapper(
            task=self.task,
            tool_name=self.tool.name,
            tool_description=self.tool.description,
            output_schema=self.tool._output.model_json_schema(),
            tool_output=input.model_dump_json(),
            response_schema=NLOutput.model_json_schema(),
            error_schema=LLMError.model_json_schema(),
            examples=self._example_outputs,
        )
        return [{"role": "user", "content": prompt}]

    def _format_chunk_messages(self, chunk: str) -> list[ChatCompletionMessageParam]:
        """Format messages for chunk simplification."""
        prompt = _templates.simplify_chunk(
            task=self.task,
            tool_name=self.tool.name,
            chunk=chunk,
            response_schema=NLOutput.model_json_schema(),
            error_schema=LLMError.model_json_schema(),
        )
        return [{"role": "user", "content": prompt}]

    def _simplify_chunk_sync(self, chunk: str) -> NLOutput:
        """Simplify a single chunk synchronously."""
        messages = self._format_chunk_messages(chunk)
        response = self.llm_client.generate(
            messages=messages,
            model=self._model,
            mode="pydantic",
            response_model=NLOutput,
        )
        return t.cast(NLOutput, response.parsed)

    async def _simplify_chunk_async(self, chunk: str) -> NLOutput:
        """Simplify a single chunk asynchronously."""
        messages = self._format_chunk_messages(chunk)
        response = await self.llm_client.agenerate(
            messages=messages,
            model=self._model,
            mode="pydantic",
            response_model=NLOutput,
        )
        return t.cast(NLOutput, response.parsed)

    def simplify(self, input: ToolOutputT) -> NLOutput | ToolOutputT:
        """
        Simplify tool output with token-aware processing.

        Returns raw output if small, simplifies if medium, raises if too large.
        """
        output_json = input.model_dump_json()
        token_count = self._count_tokens(output_json)

        # Too large - reject
        if token_count > self.token_upper_bound:
            raise OutputTooLargeError(
                self.tool.name, token_count, self.token_upper_bound
            )

        # Small enough - pass through
        if token_count <= self.token_lower_bound:
            return input

        # Medium size - simplify (possibly with chunking)
        if token_count <= self.chunk_size:
            # Single simplification call
            return self.invoke(input)

        # Large - chunk and simplify
        splitter = self._get_text_splitter()
        chunks = splitter.split_text(output_json)

        # Simplify each chunk
        simplified_parts: list[str] = []
        for chunk in chunks:
            result = self._simplify_chunk_sync(chunk)
            if not result.success or result.error:
                raise ChunkSimplificationError(
                    "Failed to simplify chunk",
                    result.error
                    or LLMError(
                        error="Unknown error",
                        type_of_error="Simplification failure",
                        content="Chunk simplification failed",
                        suggested_fix="Try with smaller chunks",
                    ),
                )
            if result.result:
                simplified_parts.append(result.result)

        # Combine and simplify again
        combined = "\n".join(simplified_parts)
        combined_result = self._simplify_chunk_sync(combined)

        return combined_result

    async def asimplify(self, input: ToolOutputT) -> NLOutput | ToolOutputT:
        """
        Async simplify tool output with token-aware processing.

        Returns raw output if small, simplifies if medium, raises if too large.
        Supports parallel chunk processing when parallel_chunks=True.
        """
        output_json = input.model_dump_json()
        token_count = self._count_tokens(output_json)

        # Too large - reject
        if token_count > self.token_upper_bound:
            raise OutputTooLargeError(
                self.tool.name, token_count, self.token_upper_bound
            )

        # Small enough - pass through
        if token_count <= self.token_lower_bound:
            return input

        # Medium size - simplify (possibly with chunking)
        if token_count <= self.chunk_size:
            return await self.ainvoke(input)

        # Large - chunk and simplify
        splitter = self._get_text_splitter()
        chunks = splitter.split_text(output_json)

        # Simplify chunks (parallel or sequential)
        if self.parallel_chunks:
            results = await asyncio.gather(
                *[self._simplify_chunk_async(chunk) for chunk in chunks]
            )
        else:
            results = [await self._simplify_chunk_async(chunk) for chunk in chunks]

        # Check for errors and collect results
        simplified_parts: list[str] = []
        for result in results:
            if not result.success or result.error:
                raise ChunkSimplificationError(
                    "Failed to simplify chunk",
                    result.error
                    or LLMError(
                        error="Unknown error",
                        type_of_error="Simplification failure",
                        content="Chunk simplification failed",
                        suggested_fix="Try with smaller chunks",
                    ),
                )
            if result.result:
                simplified_parts.append(result.result)

        # Combine and simplify again
        combined = "\n".join(simplified_parts)
        combined_result = await self._simplify_chunk_async(combined)

        return combined_result

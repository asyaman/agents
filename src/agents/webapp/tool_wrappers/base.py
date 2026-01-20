"""
Base classes for Chainlit tool wrappers.

Tool wrappers replace standard tools with Chainlit UI versions.
They maintain the same interface (input/output types) but provide
interactive UI elements instead of mock responses.
"""

from abc import abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel

from agents.tools_core.base_tool import BaseTool


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ChainlitToolWrapper(BaseTool[InputT, OutputT], Generic[InputT, OutputT]):
    """
    Base class for Chainlit tool wrappers.

    Subclasses should:
    1. Accept the original tool in __init__
    2. Override ainvoke() to provide Chainlit UI
    3. Return the same output type as the original tool
    """

    def __init__(self, original_tool: BaseTool[InputT, OutputT]):
        """
        Initialize wrapper with original tool.

        Args:
            original_tool: The tool being wrapped
        """
        super().__init__()
        self._original_tool = original_tool
        # Copy metadata from original
        self._name = original_tool.name
        self.description = original_tool.description

    def invoke(self, input: InputT) -> OutputT:
        """
        Sync invoke - typically delegates to original or raises.

        Override if sync execution is needed.
        """
        return self._original_tool.invoke(input)

    @abstractmethod
    async def ainvoke(self, input: InputT) -> OutputT:
        """
        Async invoke with Chainlit UI.

        Must be implemented by subclasses to provide UI interaction.
        """
        pass

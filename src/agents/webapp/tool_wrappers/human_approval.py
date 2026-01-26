"""
Chainlit-aware HumanApproval tool wrapper.

This wrapper replaces the mock/default HumanApproval behavior with
actual Chainlit UI elements (action buttons, text input).

Instead of auto-responding, it:
1. Shows the drafted email in a message
2. Presents Approve/Retry/Cancel buttons
3. If Retry, asks for feedback via text input
4. Returns the user's actual decision

Usage:
    # In AgentConfig
    tool_wrappers = {
        "human_approval": ChainlitHumanMailContentApproval,
    }

    # The runner will automatically wrap the tool:
    # original_tool = HumanApproval()
    # wrapped_tool = ChainlitHumanMailContentApproval(original_tool)
"""

import chainlit as cl
from loguru import logger

from agents.tools_core.base_tool import BaseTool
from agents.example_agents.email_outreach.tools import (
    HumanApprovalInput,
    HumanApprovalOutput,
)


class ChainlitHumanMailContentApproval(
    BaseTool[HumanApprovalInput, HumanApprovalOutput]
):
    """
    Chainlit wrapper for HumanApproval tool.

    Shows interactive buttons in Chainlit UI instead of mock responses.
    """

    _name = "human_approval"
    description = (
        "Requests human approval for the drafted email content via Chainlit UI."
    )
    _input = HumanApprovalInput
    _output = HumanApprovalOutput

    def __init__(self, original_tool: BaseTool | None = None):
        """
        Initialize the wrapper.

        Args:
            original_tool: The original HumanApproval tool (kept for reference)
        """
        super().__init__()
        self._original_tool = original_tool

    def invoke(self, input: HumanApprovalInput) -> HumanApprovalOutput:
        """Sync invoke - not supported for Chainlit UI."""
        raise NotImplementedError(
            "ChainlitHumanMailContentApproval requires async execution. Use ainvoke()."
        )

    async def ainvoke(self, input: HumanApprovalInput) -> HumanApprovalOutput:
        """
        Show approval UI in Chainlit and wait for user response.

        Displays:
        1. The drafted email content
        2. Action buttons: Approve, Retry, Cancel

        If user selects Retry, prompts for feedback.

        Returns:
            HumanApprovalOutput with user's decision
        """
        logger.info(f"Requesting human approval for email to: {input.email_to}")

        # Display the drafted email
        elements = [
            cl.Text(
                name="Email Draft",
                content=f"**To:** {input.email_to}\n\n---\n\n{input.email_draft}",
                display="inline",
            )
        ]

        if input.company_description:
            elements.append(
                cl.Text(
                    name="Company Info",
                    content=f"**Company:** {input.company_description[:200]}...",
                    display="side",
                )
            )

        await cl.Message(
            content="📧 **Email Draft Ready for Review**",
            elements=elements,
        ).send()

        # Show action buttons
        response = await cl.AskActionMessage(
            content="What would you like to do with this draft?",
            actions=[
                cl.Action(
                    name="approve",
                    payload={"value": "approve"},
                    label="✅ Approve & Send",
                ),
                cl.Action(
                    name="retry",
                    payload={"value": "retry"},
                    label="🔄 Revise Draft",
                ),
                cl.Action(
                    name="stop",
                    payload={"value": "stop"},
                    label="❌ Cancel",
                ),
            ],
            timeout=300,  # 5 minutes timeout
        ).send()

        # Handle timeout
        if not response:
            logger.warning("Human approval timed out")
            return HumanApprovalOutput(
                command="stop",
                draft_instructions="Approval request timed out",
            )

        # Extract selected action
        command = response.get("payload", {}).get("value", "stop")
        logger.info(f"Human selected: {command}")

        # If retry, ask for feedback
        draft_instructions = None
        if command == "retry":
            feedback_response = await cl.AskUserMessage(
                content="📝 Please provide instructions for revising the draft:",
                timeout=300,
            ).send()

            if feedback_response:
                draft_instructions = feedback_response.get("output", "")
                logger.info(f"Revision instructions: {draft_instructions[:50]}...")

        return HumanApprovalOutput(
            command=command,
            draft_instructions=draft_instructions,
        )

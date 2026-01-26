"""
Email Outreach Agent using AgentTool with email drafting and sending tools.

This module provides a pre-configured agent for handling email outreach
workflows including drafting, approval, and sending.
"""

from agents.agent_tool.agent_tool import AgentTool, AgentToolInput, AgentToolOutput
from agents.agent_tool.base_strategy import PlanningStrategy
from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool
from agents.tools.tavily import TavilySearch, TavilyInput, TavilyOutput

from agents.example_agents.email_outreach.tools import (
    DraftMail,
    DraftMailInput,
    DraftMailOutput,
    ExtractDomain,
    ExtractDomainInput,
    ExtractDomainOutput,
    HumanApproval,
    HumanApprovalInput,
    HumanApprovalOutput,
    SendEmail,
    SendEmailInput,
    SendEmailOutput,
    SummariseWebScrapeContent,
    SummariseWebScrapeContentInput,
    SummariseWebScrapeContentOutput,
)


# Guidance messages for email outreach workflow
EMAIL_OUTREACH_GUIDANCE_MESSAGES = [
    """EMAIL OUTREACH WORKFLOW:
1. Extract domain from the prospect's email address
2. If a valid company domain is found:
   - Use tavily_search to find information about the company
   - Summarize the search results to understand what the company does
3. Draft a personalized email based on the company information
4. Request human approval for the draft
5. Based on approval response:
   - 'approve': Send the email
   - 'retry': Redraft with the provided instructions
   - 'stop': Halt the workflow and finish""",
    """IMPORTANT RULES:
- Draft one e-mail at a time
- Always extract the domain first to check if it's a company email
- Use tavily_search to research the company before drafting
- If the email is from a common provider (gmail, yahoo, etc.), draft without company info
- Never send an email without human approval
- If human requests a retry, incorporate their feedback in the redraft
- After 3 retry attempts, consider stopping to avoid spam behavior""",
]


def create_email_outreach_tools(
    llm_client: LLMClient,
    tavily_max_results: int = 3,
) -> list[BaseTool]:
    """
    Create all email outreach tools.

    Args:
        llm_client: LLM client for tools that require LLM capabilities
        tavily_max_results: Max results to return from Tavily search (default 3)

    Returns:
        List of configured tools for email outreach
    """
    return [
        ExtractDomain(),
        TavilySearch(max_results=tavily_max_results),
        SummariseWebScrapeContent(llm_client=llm_client),
        DraftMail(llm_client=llm_client),
        HumanApproval(),
        SendEmail(),
    ]


def create_email_outreach_agent(
    strategy: PlanningStrategy,
    llm_client: LLMClient,
    guidance_messages: list[str] | None = None,
    parallel_tool_calls: bool = False,
    include_finish_tool: bool = True,
) -> AgentTool:
    """
    Create an email outreach agent.

    Args:
        strategy: Planning strategy for the agent (e.g., DirectStrategy, ReactStrategy)
        llm_client: LLM client for tools that require LLM capabilities
        guidance_messages: Custom guidance messages (defaults to EMAIL_OUTREACH_GUIDANCE_MESSAGES)
        parallel_tool_calls: Allow parallel tool execution (default False for sequential workflow)
        include_finish_tool: Include the finish tool (default True)

    Returns:
        Configured AgentTool for email outreach
    """
    tools = create_email_outreach_tools(llm_client)

    agent = AgentTool(
        tools=tools,
        strategy=strategy,
        guidance_messages=guidance_messages or EMAIL_OUTREACH_GUIDANCE_MESSAGES,
        parallel_tool_calls=parallel_tool_calls,
        include_finish_tool=include_finish_tool,
    )
    return agent


__all__ = [
    "create_email_outreach_agent",
    "create_email_outreach_tools",
    "EMAIL_OUTREACH_GUIDANCE_MESSAGES",
    "AgentToolInput",
    "AgentToolOutput",
    # Tool classes
    "DraftMail",
    "DraftMailInput",
    "DraftMailOutput",
    "ExtractDomain",
    "ExtractDomainInput",
    "ExtractDomainOutput",
    "HumanApproval",
    "HumanApprovalInput",
    "HumanApprovalOutput",
    "SendEmail",
    "SendEmailInput",
    "SendEmailOutput",
    "SummariseWebScrapeContent",
    "SummariseWebScrapeContentInput",
    "SummariseWebScrapeContentOutput",
    "TavilySearch",
    "TavilyInput",
    "TavilyOutput",
]

import random
import re
import typing as t
from collections.abc import Awaitable, Callable

from jinja2 import StrictUndefined, Template
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agents.llm_core.llm_client import LLMClient
from agents.tools_core.base_tool import BaseTool
from agents.tools_core.llm_base_tool import LLMTool


# =============================================================================
# SummariseWebScrapeContent Tool
# =============================================================================


class SummariseWebScrapeContentInput(BaseModel):
    message: str = Field(description="The scraped web content to summarize")


class SummariseWebScrapeContentOutput(BaseModel):
    answer: str = Field(description="The summarized content")


class SummariseWebScrapeContent(
    LLMTool[SummariseWebScrapeContentInput, SummariseWebScrapeContentOutput]
):
    _name = "summarise_web_content_of_company"
    description = "Summarizes scraped web content, highlighting the company's nature, services, products, and offerings."
    _input = SummariseWebScrapeContentInput
    _output = SummariseWebScrapeContentOutput
    example_inputs = [SummariseWebScrapeContentInput(message="...")]
    example_outputs = [SummariseWebScrapeContentOutput(answer="...")]

    def __init__(self, llm_client: LLMClient, model: str | None = None) -> None:
        super().__init__(llm_client, model)

    def format_messages(
        self, input: SummariseWebScrapeContentInput
    ) -> list[ChatCompletionMessageParam]:
        return [
            {
                "role": "system",
                "content": "You are a helpful website content summarizer.",
            },
            {
                "role": "user",
                "content": f"""You will be passed the content of a scraped company website. Please summarize it in 250-300 words focusing on what kind of company this is, the services they offer and how they operate.

Here is the content:

{input.message}""",
            },
        ]


# =============================================================================
# DraftMail Tool
# =============================================================================


class DraftMailInput(BaseModel):
    contact_name: str = Field(
        description="Name of the company employee receiving the mail"
    )
    email_address: str = Field(
        description="Email address of the company employee receiving the mail"
    )
    company_description: str = Field(
        description="A brief overview of the company for which the mail is being drafted"
    )
    previous_draft: t.Optional[str] = Field(
        default=None,
        description="Content of a previously drafted mail, if available",
    )
    draft_instructions: t.Optional[str] = Field(
        default=None,
        description="Human input on redrafting direction, applicable solely to the redraft case when a content of a previously drafted mail is provided",
    )


class DraftMailOutput(BaseModel):
    draft: str = Field(description="Email draft content")


DRAFT_MAIL_TEMPLATE = """You are a helpful sales expert, great at writing enticing emails.
You will write an email for {{sender_name}} who wants to reach out to a new prospect {{ contact_name }} who left the email address: {{email_address}}. {{sender_name}} works for the following company:
{{sender_company_desc}}.
Write no more than 300 words.
Include no placeholders! Your response should be nothing but the pure email body!
End the email with the following signature, and do not add anything else:
Best regards,
{{sender_name}}"

{%- if domain_known %}
It must be tailored as much as possible to the prospect\'s company based on the website information we fetched. Don\'t mention that we got the information from the website.
Here is company website summary:
{{company_description}}
{%- else %}
No additional information found about the prospect.
{%- endif %}

{%- if previous_draft %}
We have already contacted previously this company. Here is the previous mail content:
{{previous_draft}}
{%- endif %}
{%- if draft_instructions %}
Update the previous mail content with the folliwng instructions:
{{draft_instructions}}
{%- endif %}


"""


class DraftMail(LLMTool[DraftMailInput, DraftMailOutput]):
    _name = "draft_mail"
    description = "Write or refine sales emails tailored to companies using their provided descriptions."
    _input = DraftMailInput
    _output = DraftMailOutput
    example_inputs = [
        DraftMailInput(
            contact_name="...",
            email_address="example@mail.com",
            company_description="...",
            previous_draft="...",
            draft_instructions="...",
        ),
        DraftMailInput(
            contact_name="...",
            email_address="example@mail.com",
            company_description="...",
            previous_draft=None,
            draft_instructions=None,
        ),
    ]
    example_outputs = [DraftMailOutput(draft="...")]
    _template = DRAFT_MAIL_TEMPLATE

    def __init__(self, llm_client: LLMClient, model: str | None = None) -> None:
        super().__init__(llm_client, model)

    def format_messages(
        self, input: DraftMailInput
    ) -> list[ChatCompletionMessageParam]:
        domain_known = bool(input.company_description)
        sender_name = "agents Team"
        sender_company_desc = """agents is an AI Agent automation platform offering:

- Simplicity: Interaction with the system in human language
- Scalability: Being able to integrate tools and data with minimal effort compared to other solutions.
- Self-design: A system can design its own process and learn from it and improve itself over time...
 """

        template = Template(self._template, undefined=StrictUndefined)
        rendered_template = template.render(
            sender_name=sender_name,
            sender_company_desc=sender_company_desc,
            email_address=input.email_address,
            domain_known=domain_known,
            contact_name=input.contact_name,
            company_description=input.company_description,
            previous_draft=input.previous_draft if input.previous_draft else None,
            draft_instructions=(
                input.draft_instructions if input.draft_instructions else None
            ),
        )

        return [
            {"role": "user", "content": rendered_template},
        ]


# =============================================================================
# ExtractDomain Tool
# =============================================================================


class ExtractDomainInput(BaseModel):
    email_address: str = Field(description="The email address to extract domain from")


class ExtractDomainOutput(BaseModel):
    lead_website_url: t.Optional[str] = Field(
        description="The extracted domain URL for scraping"
    )


class ExtractDomain(BaseTool[ExtractDomainInput, ExtractDomainOutput]):
    _name = "url_extraction"
    description = (
        "Extract url from provided e-mail address. Will return only the domain url"
    )
    _input = ExtractDomainInput
    _output = ExtractDomainOutput
    example_inputs = [ExtractDomainInput(email_address="example@firecrawl.com")]
    example_outputs = [ExtractDomainOutput(lead_website_url="https://firecrawl.com")]

    def invoke(self, input: ExtractDomainInput) -> ExtractDomainOutput:
        email_address = input.email_address
        common_providers = [
            "gmail",
            "yahoo",
            "ymail",
            "rocketmail",
            "outlook",
            "hotmail",
            "live",
            "msn",
            "icloud",
            "me",
            "mac",
            "aol",
            "zoho",
            "protonmail",
            "mail",
            "gmx",
        ]

        url = None
        domain = email_address.split("@")[-1] if "@" in email_address else None
        if domain:
            pattern = rf"^({'|'.join(common_providers)})\.(?:com|net|org|edu|.*)"
            if not re.search(pattern, domain):
                url = f"https://{domain}"
        print(f"Extracted domain: {url}")
        return ExtractDomainOutput(lead_website_url=url)


# =============================================================================
# HumanApproval Tool
# =============================================================================


class HumanApprovalInput(BaseModel):
    email_draft: str = Field(description="The content of the drafted email")
    email_to: str = Field(
        description="Email address of the company employee who will receive the email"
    )
    company_description: str = Field(
        description="A brief description of the company for which the email is being drafted"
    )


class HumanApprovalOutput(BaseModel):
    command: str = Field(
        description="The human decision on the drafted content, either 'approve', 'retry', 'stop'."
    )
    draft_instructions: t.Optional[str] = Field(
        default=None,
        description="Human input on redrafting direction, applicable solely to the 'retry' option i.e only when content requires redrafting",
    )


# Type aliases for approval functions
ApprovalFn = Callable[[HumanApprovalInput], HumanApprovalOutput]
AsyncApprovalFn = Callable[[HumanApprovalInput], Awaitable[HumanApprovalOutput]]


class MockApprovalStrategy:
    """
    Mock approval strategy for testing.
    Simulates human behavior: first request returns retry, then random, then stop.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._call_count = 0
        self._max_retries = max_retries

    def __call__(self, input: HumanApprovalInput) -> HumanApprovalOutput:
        self._call_count += 1

        if self._call_count == 1:
            command = "retry"
        elif self._call_count > self._max_retries:
            command = "stop"
        else:
            command = random.choice(["approve", "retry"])

        draft_instructions = "Redraft more friendly" if command == "retry" else None
        print(f"Human request at attempt {self._call_count}: {command}")

        return HumanApprovalOutput(
            command=command, draft_instructions=draft_instructions
        )

    def reset(self) -> None:
        """Reset the call counter."""
        self._call_count = 0


class HumanApproval(BaseTool[HumanApprovalInput, HumanApprovalOutput]):
    """
    Requests human approval for drafted email content.

    This tool supports pluggable approval strategies:
    - Pass a sync function for custom approval logic
    - Pass an async function for real human-in-the-loop (UI, Slack, etc.)
    - Use default MockApprovalStrategy for testing

    Example:
        # Mock for testing
        tool = HumanApproval()

        # Custom sync approval
        tool = HumanApproval(approval_fn=lambda inp: HumanApprovalOutput(command="approve"))

        # Async approval (e.g., wait for Slack response)
        async def slack_approval(inp: HumanApprovalInput) -> HumanApprovalOutput:
            response = await wait_for_slack_response(inp.email_draft)
            return HumanApprovalOutput(command=response.command, ...)
        tool = HumanApproval(async_approval_fn=slack_approval)
    """

    _name = "human_approval"
    description = "Requests human approval for the drafted email content. Returns either 'approve', 'retry' or 'stop'."
    _input = HumanApprovalInput
    _output = HumanApprovalOutput
    example_inputs = [
        HumanApprovalInput(
            email_draft="...", email_to="example@gmail.com", company_description="..."
        )
    ]
    example_outputs = [
        HumanApprovalOutput(command="approve", draft_instructions=None),
        HumanApprovalOutput(command="retry", draft_instructions="..."),
        HumanApprovalOutput(command="stop", draft_instructions=None),
    ]

    def __init__(
        self,
        approval_fn: ApprovalFn | None = None,
        async_approval_fn: AsyncApprovalFn | None = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the HumanApproval tool.

        Args:
            approval_fn: Sync approval function. If None, uses MockApprovalStrategy.
            async_approval_fn: Async approval function for real human-in-the-loop.
            max_retries: Max retries for MockApprovalStrategy (only used if approval_fn is None).
        """
        self._mock_strategy = MockApprovalStrategy(max_retries=max_retries)
        self._approval_fn = approval_fn or self._mock_strategy
        self._async_approval_fn = async_approval_fn

    def invoke(self, input: HumanApprovalInput) -> HumanApprovalOutput:
        """Execute sync approval."""
        return self._approval_fn(input)

    async def ainvoke(self, input: HumanApprovalInput) -> HumanApprovalOutput:
        """Execute async approval if available, otherwise fall back to sync."""
        if self._async_approval_fn is not None:
            return await self._async_approval_fn(input)
        return self._approval_fn(input)

    def reset(self) -> None:
        """Reset the mock strategy counter (only applies when using default mock)."""
        if isinstance(self._approval_fn, MockApprovalStrategy):
            self._mock_strategy.reset()


# =============================================================================
# SendEmail Tool
# =============================================================================


class SendEmailInput(BaseModel):
    content: str = Field(description="The content of the drafted email")
    mail_to: str = Field(
        description="Email address of the company employee who will receive the email"
    )
    mail_cc: t.Optional[str] = Field(
        default=None, description="The CC recipient email address"
    )
    mail_bc: t.Optional[str] = Field(
        default=None, description="The BCC recipient email address"
    )


class SendEmailOutput(BaseModel):
    result: str = Field(description="Confirmation of the response as successful or not")


class SendEmail(BaseTool[SendEmailInput, SendEmailOutput]):
    _name = "sendmail"
    description = "Sends mail to a provided recipient with input content"
    _input = SendEmailInput
    _output = SendEmailOutput
    example_inputs = [
        SendEmailInput(
            mail_to="valerio@gmail.com", content="...", mail_cc=None, mail_bc=None
        ),
        SendEmailInput(
            mail_to="sam@gmail.com",
            content="...",
            mail_cc="james@gmail.com",
            mail_bc=None,
        ),
    ]
    example_outputs = [SendEmailOutput(result="E-mail successfully sent")]

    def invoke(self, input: SendEmailInput) -> SendEmailOutput:
        result = "done"
        print(f"Send mail to: {input.mail_to}")
        return SendEmailOutput(result=result)

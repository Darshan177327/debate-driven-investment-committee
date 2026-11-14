
import asyncio

from dotenv import load_dotenv
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient

from tools.research_tools import (
    search_annual_report,
    semantic_search_annual_report,
    set_active_company,
    get_active_company
)

load_dotenv()


# ==================================================
# RESEARCH AGENT
# ==================================================

async def get_research_report(
    company_name: str = "Jio Financial Services"
):
    """
    Microsoft Agent Framework Research Agent.

    Researches the selected company's indexed documents
    using local FastEmbed embeddings and ChromaDB.

    Gemini is used only for agent reasoning and report generation.
    """

    print("\n==========================================")
    print(" MICROSOFT AGENT FRAMEWORK")
    print(" MULTI-COMPANY RESEARCH AGENT")
    print("==========================================")

    # Select company before starting research.
    set_active_company(company_name)

    active_company = get_active_company()

    print(f"\n[RESEARCH] Selected company: {active_company}")

    async with GeminiChatClient(
        model="gemini-3.5-flash-lite"
    ) as client:

        research_agent = Agent(
            client=client,
            name="ResearchAgent",

            instructions=f"""
You are the Research Agent of a Debate-Driven
Investment Committee.

Your assigned company is: {active_company}

You must research ONLY this company's indexed documents.

DOCUMENT RETRIEVAL ARCHITECTURE:

The documents are stored in company-specific ChromaDB
collections.

Semantic embeddings are generated locally using
BAAI/bge-small-en-v1.5 through FastEmbed.

Gemini is used for reasoning and report generation.
Do not claim that Gemini generates document embeddings.

AVAILABLE PYTHON TOOLS:

1. semantic_search_annual_report(query)

   Searches the selected company's indexed documents
   using local semantic embeddings.

2. search_annual_report(keyword)

   Performs exact keyword matching against the selected
   company's indexed document chunks.

MANDATORY TOOL WORKFLOW:

1. Call semantic_search_annual_report with a query covering:
   "{active_company} FY2025 financial performance,
   revenue, income, expenses, profit and financial risks."

2. Call search_annual_report with:
   "Total Income"

3. Use the returned evidence to prepare your report.

4. Search additional terms when needed, such as:
   "Profit for the year", "Interest Income",
   "Dividend Income", "Impairment",
   "Borrowings", and "Fair value changes".

FINANCIAL RESEARCH RULES:

- Research only {active_company}.
- Use retrieved documents as the primary evidence.
- Cite the source filename and chunk number.
- Do not invent original PDF page numbers.
- Keep standalone and consolidated figures separate.
- Never invent financial figures.
- Do not invent explanations for financial changes.
- Distinguish reported facts from interpretations.
- Do not treat missing information as proof of poor performance.
- Correctly label financial years.
- If evidence is insufficient, explicitly state so.
- Do not make a BUY, HOLD or AVOID recommendation.
- Do not confuse revenue, AUM, premiums facilitated,
  assets, profit and cash flow.

OUTPUT FORMAT:

1. Company and reporting period.
2. Standalone financial performance, if available.
3. Consolidated financial performance, if available.
4. Important income components.
5. Potential financial strengths.
6. Potential financial risks.
7. Evidence limitations.
8. Sources: filename and chunk numbers.

Prepare a concise, evidence-based research report.
""",

            tools=[
                semantic_search_annual_report,
                search_annual_report
            ]
        )

        question = f"""
Prepare an investment research report for {active_company}.

Use the indexed documents belonging to this company only.

MANDATORY ACTIONS:

1. Search the document collection semantically for
   FY2025 financial performance.
2. Search the documents for "Total Income".
3. Search additional relevant financial terms where needed.
4. Support important statements with retrieved evidence.

INVESTIGATE:

A. Standalone performance, where available:
- Total income.
- Total expenses.
- Profit before tax.
- Profit after tax.
- Interest income.
- Dividend income.
- Fair value gains.
- Fees and commission.

B. Consolidated performance, where available:
- Total income.
- Total expenses.
- Profit before tax.
- Profit after tax.
- Important income components.

C. Investment considerations:
- Positive financial indicators.
- Potential financial risks.
- Important financial changes.
- Areas requiring additional verification.

IMPORTANT:

- Do not mix standalone and consolidated figures.
- Do not assume every company has the same reporting structure.
- Do not fabricate missing financial metrics.
- Cite the exact source filename and chunk number.
- Original PDF page numbers are not currently available
  in this retrieval index.
- Clearly distinguish reported facts from your analysis.
- Do not issue an investment recommendation.

Return a structured research report for the committee.
"""

        print("\n[RESEARCH] Running Research Agent...")

        response = await research_agent.run(question)

        report = response.text

        if not report or not report.strip():
            raise RuntimeError(
                "Research Agent returned an empty report."
            )

        print("\n==========================================")
        print(" RESEARCH AGENT OUTPUT")
        print("==========================================\n")

        print(report)

        print("\n[RESEARCH] Report successfully generated.")

        return report


# ==================================================
# STANDALONE TEST
# ==================================================

async def main():

    report = await get_research_report(
        "Jio Financial Services"
    )

    print("\n==========================================")
    print(" RESEARCH AGENT EXECUTION COMPLETED")
    print("==========================================")

    return report


if __name__ == "__main__":
    asyncio.run(main())

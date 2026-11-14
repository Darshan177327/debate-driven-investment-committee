
import asyncio

from dotenv import load_dotenv
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient

from tools.research_tools import (
    semantic_search_annual_report,
    search_annual_report,
    get_active_company
)

load_dotenv()


async def _run_bear_agent(research):

    if not research or not research.strip():
        raise ValueError("Shared research report is unavailable.")

    company = get_active_company()

    print("\n========== MICROSOFT FRAMEWORK BEAR AGENT ==========")
    print(f"[BEAR] Researching: {company}")

    # -------------------------------------------------
    # STEP 1: EXECUTE TWO REAL PYTHON TOOLS
    # -------------------------------------------------

    print("\n[BEAR TOOL 1/2] Semantic RAG search")

    rag_evidence = semantic_search_annual_report(
        f"{company} financial risks, declining income, "
        "fair value gains, rising expenses, impairment, "
        "debt, asset quality and profitability sustainability."
    )

    if not rag_evidence or "No relevant evidence" in rag_evidence:
        raise RuntimeError("Bear Agent RAG search returned no evidence.")

    print("[BEAR] Semantic search completed.")

    print("\n[BEAR TOOL 2/2] Keyword search")

    keyword_evidence = search_annual_report("Impairment")

    if not keyword_evidence:
        raise RuntimeError("Bear Agent keyword search failed.")

    print("[BEAR] Keyword search completed.")
    print("[BEAR] Both mandatory Python tools executed.")

    # -------------------------------------------------
    # STEP 2: PREPARE SHARED EVIDENCE
    # -------------------------------------------------

    verified_evidence = f"""
SELECTED COMPANY:
{company}

SHARED RESEARCH REPORT:
{research}

ADDITIONAL SEMANTIC RAG EVIDENCE:
{rag_evidence}

ADDITIONAL KEYWORD SEARCH EVIDENCE:
{keyword_evidence}
"""

    # -------------------------------------------------
    # STEP 3: MICROSOFT AGENT FRAMEWORK BEAR AGENT
    # -------------------------------------------------

    async with GeminiChatClient(
        model="gemini-3.5-flash-lite"
    ) as client:

        bear_agent = Agent(
            client=client,
            name="BearAgent",

            instructions=f"""
You are the BEAR AGENT in a Debate-Driven Investment Committee.

Your assigned company is {company}.

Your responsibility is to construct the strongest
evidence-based argument AGAINST investing in this company.

You are a Microsoft Agent Framework Agent.

AVAILABLE PYTHON TOOLS:

1. semantic_search_annual_report(query)
   Searches the selected company's ChromaDB collection
   using locally generated FastEmbed embeddings.

2. search_annual_report(keyword)
   Searches indexed document chunks for exact keywords.

Both tools have already been executed by the workflow.
Their results are included in the input.

You may call them again if additional verification is needed.

RESEARCH RULES:

1. Present exactly three bearish arguments.
2. Use the shared research report and retrieved evidence.
3. Cite source filenames and chunk numbers.
4. Never invent financial figures.
5. Keep standalone and consolidated figures separate
   whenever the source provides both.
6. Analyze fair value gains and their potential volatility.
7. Examine expenses and profitability sustainability.
8. Examine credit and impairment risks where evidence exists.
9. Examine regulatory and operational risks where evidence exists.
10. Distinguish facts from interpretations.
11. Missing evidence does not prove poor performance.
12. If a metric is not established, say:
    "Not established by the retrieved evidence."
13. Historical performance does not guarantee future returns.
14. Do not make a final BUY, HOLD or AVOID decision.
15. Do not invent causal explanations for financial changes.
16. Do not call fair value gains non-operating unless
    the accounting evidence establishes that classification.
17. Do not treat AUM as revenue.
18. Do not assume rising impairment automatically proves
    widespread portfolio stress.
19. Evaluate absolute amounts as well as percentage changes.
20. Do not exaggerate risks to strengthen the bearish case.
21. Do not invent original PDF page numbers.
22. Research only {company}.

OUTPUT FORMAT:

BEARISH INVESTMENT CASE

1. Argument
   - Supporting Evidence
   - Source Filename and Chunk
   - Investment Implication

2. Argument
   - Supporting Evidence
   - Source Filename and Chunk
   - Investment Implication

3. Argument
   - Supporting Evidence
   - Source Filename and Chunk
   - Investment Implication

KEY RISKS AND UNCERTAINTIES

MISSING INFORMATION REQUIRING FURTHER INVESTIGATION

WHY AN INVESTOR MIGHT AVOID OR DELAY INVESTING
""",

            tools=[
                semantic_search_annual_report,
                search_annual_report
            ]
        )

        prompt = f"""
Prepare the bearish investment case for {company}.

Use this retrieved evidence:

{verified_evidence}

FOCUS ON:

- Consolidated versus standalone profitability.
- Changes in income components.
- Fair value gain exposure.
- Rising expenses.
- Financial instrument impairment.
- Lending and asset quality uncertainties.
- Regulatory and operational uncertainties.

YOUR TASK:

1. Produce exactly three evidence-based bearish arguments.
2. Explain the supporting financial evidence.
3. Cite source filenames and chunk numbers.
4. Explain the investment implications.
5. Acknowledge uncertainty and evidence limitations.
6. Distinguish facts from interpretations.
7. Do not exaggerate the available evidence.

Do not make the committee's final investment decision.
"""

        print("\n[BEAR] Running Microsoft Agent Framework Agent...")

        response = await bear_agent.run(prompt)

        if not response.text or not response.text.strip():
            raise RuntimeError("Bear Agent returned an empty response.")

        print("\n========== BEAR AGENT RESULT ==========\n")
        print(response.text)

        return response.text


def run_bear_agent(research):
    """
    Synchronous interface retained for the existing
    committee workflow.
    """

    try:
        return asyncio.run(_run_bear_agent(research))

    except Exception as error:
        print("[BEAR AGENT ERROR]", error)
        raise


if __name__ == "__main__":

    print(
        "Bear Agent requires a shared research report. "
        "Test it through the committee workflow."
    )

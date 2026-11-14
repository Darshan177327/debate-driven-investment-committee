
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


async def _run_bull_agent(research):

    if not research or not research.strip():
        raise ValueError("Shared research report is unavailable.")

    company = get_active_company()

    print("\n========== MICROSOFT FRAMEWORK BULL AGENT ==========")
    print(f"[BULL] Researching: {company}")

    # -------------------------------------------------
    # STEP 1: EXECUTE TWO REAL PYTHON TOOLS
    # -------------------------------------------------

    print("\n[BULL TOOL 1/2] Semantic RAG search")

    rag_evidence = semantic_search_annual_report(
        f"{company} financial strengths, revenue, "
        "profitability, financial performance, business "
        "growth and investment opportunities."
    )

    if not rag_evidence or "No relevant evidence" in rag_evidence:
        raise RuntimeError("Bull Agent could not retrieve RAG evidence.")

    print("[BULL] Semantic search completed.")

    print("\n[BULL TOOL 2/2] Keyword search")

    keyword_evidence = search_annual_report("Total Income")

    if not keyword_evidence:
        raise RuntimeError("Bull Agent keyword search failed.")

    print("[BULL] Keyword search completed.")
    print("[BULL] Both mandatory Python tools executed.")

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
    # STEP 3: MICROSOFT AGENT FRAMEWORK BULL AGENT
    # -------------------------------------------------

    async with GeminiChatClient(
        model="gemini-3.5-flash-lite"
    ) as client:

        bull_agent = Agent(
            client=client,
            name="BullAgent",

            instructions=f"""
You are the BULL AGENT in a Debate-Driven Investment Committee.

Your assigned company is {company}.

Your responsibility is to construct the strongest
evidence-based argument IN FAVOUR of investing in
this company.

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

MANDATORY RESEARCH RULES:

1. Present exactly three bullish arguments.
2. Support financial claims with retrieved evidence.
3. Cite source filenames and chunk numbers.
4. Never invent financial figures.
5. Keep standalone and consolidated figures separate
   whenever the source provides both.
6. Do not assume zero finance costs mean zero debt.
7. Do not confuse premiums facilitated with revenue.
8. Explain the opportunity and uncertainty associated
   with fair value gains.
9. Clearly distinguish reported facts from interpretations.
10. Identify assumptions requiring verification.
11. Large net worth does not automatically prove
    profitability, solvency or investment attractiveness.
12. Historical performance does not guarantee future returns.
13. Do not make a final BUY, HOLD or AVOID decision.
14. Do not invent causal explanations for financial changes.
15. Do not describe rising expenses as growth investments
    unless the evidence supports that interpretation.
16. Do not treat downloads or users as proof of profitability.
17. Do not treat AUM as company revenue.
18. Do not invent original PDF page numbers.
19. If evidence is insufficient, explicitly state so.
20. Research only {company}.

OUTPUT FORMAT:

BULLISH INVESTMENT CASE

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

ASSUMPTIONS REQUIRING VERIFICATION

ACKNOWLEDGED RISKS

WHY AN INVESTOR MIGHT CONSIDER THIS COMPANY
""",

            tools=[
                semantic_search_annual_report,
                search_annual_report
            ]
        )

        prompt = f"""
Prepare the bullish investment case for {company}.

Use the following retrieved evidence:

{verified_evidence}

YOUR TASK:

1. Identify exactly three genuine investment strengths.
2. Explain the financial evidence supporting each.
3. Cite the source filename and chunk number.
4. Explain why each strength could matter to an investor.
5. Acknowledge relevant risks and uncertainties.
6. Distinguish facts from assumptions.
7. Do not turn assumptions into reported facts.
8. Do not use evidence belonging to another company.

Do not make the committee's final investment decision.
"""

        print("\n[BULL] Running Microsoft Agent Framework Agent...")

        response = await bull_agent.run(prompt)

        if not response.text or not response.text.strip():
            raise RuntimeError("Bull Agent returned an empty response.")

        print("\n========== BULL AGENT RESULT ==========\n")
        print(response.text)

        return response.text


def run_bull_agent(research):
    """
    Synchronous interface retained for the existing
    committee workflow.
    """

    try:
        return asyncio.run(_run_bull_agent(research))

    except Exception as error:
        print("[BULL AGENT ERROR]", error)
        raise


if __name__ == "__main__":

    print(
        "Bull Agent requires a shared research report. "
        "Test it through the committee workflow."
    )

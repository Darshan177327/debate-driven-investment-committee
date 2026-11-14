
import asyncio

from dotenv import load_dotenv
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient

from agents.framework_research_agent import get_research_report
from agents.bull_agent import run_bull_agent
from agents.bear_agent import run_bear_agent

from tools.research_tools import (
    semantic_search_annual_report,
    search_annual_report,
    get_active_company,
)

load_dotenv()


async def _run_framework_rebuttal(
    research,
    bull_argument,
    bear_argument,
):
    """
    Microsoft Agent Framework Rebuttal Agent.

    Produces:
    1. Bull's response to Bear's three strongest objections.
    2. Bear's response to Bull's three strongest arguments.

    Evidence is retrieved from the selected company's local
    ChromaDB collection using FastEmbed.
    """

    company = get_active_company()

    print("\n========== MICROSOFT FRAMEWORK REBUTTAL AGENT ==========")
    print(f"[REBUTTAL] Selected company: {company}")

    # -------------------------------------------------
    # STEP 1: EXECUTE TWO REAL PYTHON EVIDENCE TOOLS
    # -------------------------------------------------

    print("\n[REBUTTAL TOOL 1/2] Semantic evidence search")

    rag_evidence = semantic_search_annual_report(
        f"{company} financial strengths and risks, profitability, "
        "income, expenses, impairment, fair value changes, "
        "capital position and business performance"
    )

    if not rag_evidence:
        raise RuntimeError(
            "Rebuttal Agent semantic search returned no evidence."
        )

    print("[REBUTTAL] Semantic evidence search completed.")

    print("\n[REBUTTAL TOOL 2/2] Keyword evidence search")

    keyword_evidence = search_annual_report("Impairment")

    if not keyword_evidence:
        keyword_evidence = (
            "No matching impairment keyword evidence was found. "
            "Do not infer that impairment is zero."
        )

    print("[REBUTTAL] Keyword evidence search completed.")
    print("[REBUTTAL] Both Python evidence tools executed.")

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
    # STEP 3: CREATE MICROSOFT AGENT FRAMEWORK AGENT
    # -------------------------------------------------

    async with GeminiChatClient(
        model="gemini-3.5-flash-lite"
    ) as client:

        rebuttal_agent = Agent(
            client=client,
            name="RebuttalAgent",

            instructions="""
You are the REBUTTAL AGENT in a Debate-Driven Investment
Committee.

Your responsibility is to conduct one fair rebuttal turn
for each side of the investment debate.

You must produce TWO separate sections.

PART A — BULL AGENT REBUTTAL:
The Bull responds to the Bear's three strongest objections.

PART B — BEAR AGENT REBUTTAL:
The Bear responds to the Bull's three strongest arguments.

EVIDENCE TOOLS:

1. semantic_search_annual_report(query)
   Searches the selected company's local ChromaDB collection.

2. search_annual_report(keyword)
   Searches indexed report chunks for matching terms.

Both tools have already been executed. Their results are
included in the input. You may call them again if needed.

IMPORTANT EVIDENCE RULES:

1. Give each side exactly one rebuttal turn.
2. Address exactly three opposing arguments per side.
3. Use only evidence supported by the supplied research,
   debate arguments or retrieved report chunks.
4. Cite the source filename and chunk number when available.
5. Do not invent original PDF page numbers.
6. Do not invent financial figures.
7. Keep standalone and consolidated figures separate.
8. Distinguish reported facts from interpretations.
9. Acknowledge valid arguments made by the opposing side.
10. Identify unsupported assumptions and explain why the
    available evidence does not establish them.
11. Do not simply repeat the original arguments.
12. Do not make a final BUY, HOLD or AVOID recommendation.
13. Do not decide which side wins the debate.
14. Do not treat a large net worth as proof of profitability
    or guaranteed solvency.
15. Do not assume rising expenses are growth investments
    without supporting evidence.
16. Do not assume impairment increases prove widespread
    credit stress.
17. Do not describe fair value gains as non-operating unless
    the accounting evidence establishes that classification.
18. Do not claim a regulatory or corporate restructuring
    caused a financial change without supporting evidence.
19. Do not treat AUM, app downloads or facilitated premiums
    as company revenue.
20. Missing evidence does not prove poor performance.
21. If evidence is insufficient, state that clearly.
22. Do not mix evidence from another company's collection.
23. Do not present a ChromaDB chunk number as a PDF page.

PART A OUTPUT FORMAT:

========== BULL AGENT REBUTTAL ==========

Objection 1:
Bear's claim:
Bull's response:
Supporting evidence (filename and chunk):

Objection 2:
Bear's claim:
Bull's response:
Supporting evidence (filename and chunk):

Objection 3:
Bear's claim:
Bull's response:
Supporting evidence (filename and chunk):

Strongest unresolved bearish concern:
Bull's final response:

PART B OUTPUT FORMAT:

========== BEAR AGENT REBUTTAL ==========

Bull argument 1:
Bull's claim:
Bear's response:
Supporting evidence (filename and chunk):

Bull argument 2:
Bull's claim:
Bear's response:
Supporting evidence (filename and chunk):

Bull argument 3:
Bull's claim:
Bear's response:
Supporting evidence (filename and chunk):

Strongest unresolved bullish argument:
Bear's final response:
""",

            tools=[
                semantic_search_annual_report,
                search_annual_report,
            ],
        )

        # -------------------------------------------------
        # STEP 4: GIVE BOTH SIDES THE SAME EVIDENCE
        # -------------------------------------------------

        prompt = f"""
Generate both rebuttal sections for the selected company.

{verified_evidence}

==================================================
INITIAL BULLISH ARGUMENT
==================================================

{bull_argument}

==================================================
INITIAL BEARISH ARGUMENT
==================================================

{bear_argument}

TASK A — BULL REBUTTAL:

Respond to the Bear's three strongest objections.

Acknowledge valid risks, challenge weak assumptions and
support responses with retrieved evidence.

TASK B — BEAR REBUTTAL:

Respond to the Bull's three strongest arguments.

Acknowledge valid strengths, challenge weak assumptions and
support responses with retrieved evidence.

Both sides must identify their strongest unresolved
opposing argument.

FINAL REQUIREMENTS:

- Exactly three objections in each rebuttal.
- Cite source filename and chunk number when available.
- Do not invent PDF page numbers.
- Keep standalone and consolidated data separate.
- Do not invent financial figures or causal explanations.
- Do not make the final investment decision.
- Do not omit either side's rebuttal.
"""

        print("\n[REBUTTAL] Running Microsoft Agent Framework Agent...")

        response = await rebuttal_agent.run(prompt)

        if not response.text or not response.text.strip():
            raise RuntimeError(
                "Rebuttal Agent returned an empty response."
            )

        print("\n========== COMPLETE REBUTTAL RESULT ==========\n")
        print(response.text)

        return response.text


# -----------------------------------------------------
# MAIN SYNCHRONOUS COMMITTEE INTERFACE
# -----------------------------------------------------

def run_rebuttal_agent(company_name="Jio Financial Services"):

    print("\n" + "=" * 55)
    print(" RESEARCH AND DEBATE ROUND")
    print("=" * 55)

    print(f"\nSelected company: {company_name}")

    # STEP 1: Run Research Agent exactly once.

    print("\nSTEP 1: RUNNING RESEARCH AGENT")

    try:
        research = asyncio.run(
            get_research_report(company_name)
        )

    except Exception as error:
        print("Research Agent execution failed:", error)
        return None

    if not research:
        print("Research unavailable. Debate cannot continue.")
        return None

    # STEP 2: Run Bull Agent using the shared research.

    print("\nSTEP 2: RUNNING BULL AGENT")

    try:
        bull_argument = run_bull_agent(research)
    except Exception as error:
        print("Bull Agent execution failed:", error)
        return None

    if not bull_argument:
        print("Bull Agent failed. Debate cannot continue.")
        return None

    # STEP 3: Run Bear Agent using the same research.

    print("\nSTEP 3: RUNNING BEAR AGENT")

    try:
        bear_argument = run_bear_agent(research)
    except Exception as error:
        print("Bear Agent execution failed:", error)
        return None

    if not bear_argument:
        print("Bear Agent failed. Debate cannot continue.")
        return None

    # STEP 4: Run Rebuttal Agent.

    print("\nSTEP 4: RUNNING REBUTTAL AGENT")

    try:
        rebuttal_result = asyncio.run(
            _run_framework_rebuttal(
                research,
                bull_argument,
                bear_argument,
            )
        )

    except Exception as error:
        print("Rebuttal Agent execution failed:", error)
        return None

    if not rebuttal_result:
        print("Rebuttal Agent failed. Debate cannot continue.")
        return None

    # STEP 5: Assemble complete debate transcript.

    debate = f"""
==================================================
DEBATE-DRIVEN INVESTMENT COMMITTEE
COMPANY: {company_name}
==================================================

SHARED RESEARCH REPORT
{research}

==================================================
INITIAL BULLISH ARGUMENT
==================================================
{bull_argument}

==================================================
INITIAL BEARISH ARGUMENT
==================================================
{bear_argument}

==================================================
BULL AND BEAR REBUTTALS
==================================================
{rebuttal_result}

==================================================
END OF DEBATE TRANSCRIPT
==================================================
"""

    print("\n" + "=" * 55)
    print(" COMPLETE DEBATE TRANSCRIPT GENERATED")
    print("=" * 55)

    return debate


if __name__ == "__main__":

    result = run_rebuttal_agent()

    if result:
        print("\nDebate execution completed successfully.")
    else:
        print("\nDebate execution failed.")

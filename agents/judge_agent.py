
import asyncio

from dotenv import load_dotenv
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient

from agents.rebuttal_agent import run_rebuttal_agent

from tools.research_tools import (
    semantic_search_annual_report,
    search_annual_report,
    get_active_company,
)

load_dotenv()


# =====================================================
# MICROSOFT AGENT FRAMEWORK JUDGE
# =====================================================

async def _run_framework_judge(debate):

    if not debate or not debate.strip():
        raise ValueError("Debate transcript is unavailable.")

    company = get_active_company()

    print("\n========== MICROSOFT FRAMEWORK JUDGE AGENT ==========")
    print(f"[JUDGE] Selected company: {company}")
    print("[JUDGE] Starting evidence verification...")

    # -------------------------------------------------
    # STEP 1: EXECUTE TWO REAL PYTHON EVIDENCE TOOLS
    # -------------------------------------------------

    print("\n[JUDGE TOOL 1/2] Semantic RAG evidence search")

    rag_evidence = semantic_search_annual_report(
        f"{company} financial performance, revenue, income, "
        "expenses, profitability, cash flow, assets, liabilities, "
        "debt, impairment, business growth and financial risks"
    )

    if not rag_evidence:
        raise RuntimeError(
            "Judge Agent semantic search returned no evidence."
        )

    print("[JUDGE] Semantic evidence search completed.")

    print("\n[JUDGE TOOL 2/2] Keyword evidence search")

    keyword_evidence = search_annual_report("Impairment")

    if not keyword_evidence:
        keyword_evidence = (
            "No matching impairment keyword evidence was found. "
            "This does not establish that impairment is zero."
        )

    print("[JUDGE] Keyword evidence search completed.")
    print("[JUDGE] Both Python evidence tools executed successfully.")

    # -------------------------------------------------
    # STEP 2: PREPARE EVIDENCE
    # -------------------------------------------------

    verified_evidence = f"""
SELECTED COMPANY:
{company}

ADDITIONAL SEMANTIC RAG EVIDENCE:
{rag_evidence}

ADDITIONAL KEYWORD SEARCH EVIDENCE:
{keyword_evidence}
"""

    # -------------------------------------------------
    # STEP 3: CREATE MICROSOFT FRAMEWORK JUDGE AGENT
    # -------------------------------------------------

    async with GeminiChatClient(
        model="gemini-3.5-flash-lite"
    ) as client:

        judge_agent = Agent(
            client=client,
            name="JudgeAgent",

            instructions="""
You are the JUDGE AGENT of a Debate-Driven Investment Committee.

Your responsibility is to objectively evaluate the Bull Agent
and Bear Agent after their rebuttal round.

You are implemented using Microsoft Agent Framework.

==================================================
AVAILABLE PYTHON TOOLS
==================================================

1. semantic_search_annual_report(query)

   Searches the selected company's local ChromaDB collection.

2. search_annual_report(keyword)

   Searches indexed document chunks for matching terms.

Both tools have already been executed by the Judge workflow.
Their results are included in the input.

You may call these tools again if additional verification
is required.

==================================================
DECISION RULES
==================================================

1. Evaluate both arguments impartially.
2. Judge evidence quality, not persuasive language.
3. Give importance to verified source-document facts.
4. Consider separate financial statements separately from
   consolidated financial statements.
5. Never invent financial figures or facts.
6. Do not treat assumptions as established facts.
7. Consider both sides' rebuttals.
8. Identify unresolved disagreements.
9. Identify limitations in the available evidence.
10. Do not automatically favour either side.
11. Choose exactly one recommendation:
    BUY, HOLD, or AVOID.
12. Explain why the chosen recommendation is appropriate.
13. This is an academic investment analysis, not financial advice.
14. Evaluate only the company identified in the input.
15. Do not mix evidence from another company's collection.

==================================================
STRICT EVIDENCE AUDIT
==================================================

For every important Bull and Bear claim, classify it as:

1. VERIFIED FACT:

   Directly supported by the supplied source documents.
   Cite the filename and chunk number when available.

2. REASONABLE INFERENCE:

   A logical interpretation of verified facts.
   Explicitly label it as an inference.

3. UNSUPPORTED CLAIM:

   A claim that cannot be established from the available evidence.
   Do not use it as a deciding reason.

Apply the same evidence standards to both sides.

Explicitly identify unsupported claims made during rebuttals.

Do not invent original PDF page numbers.
A ChromaDB chunk number is not a PDF page number.

==================================================
FINANCIAL ACCURACY REQUIREMENTS
==================================================

1. Verify financial figures against retrieved evidence.
2. Compare financial periods only when the documents
   provide comparable figures.
3. Keep standalone and consolidated results separate.
4. Do not compare different reporting periods as if they
   were the same period.
5. Distinguish revenue, other income, gains and profit.
6. Distinguish operating cash flow from accounting profit.
7. Distinguish assets, liabilities, borrowings and equity.
8. Do not confuse AUM, customer count, downloads or
   transaction volume with company revenue.
9. Do not invent financial ratios.
10. If calculating a ratio, show the figures and formula.
11. If a figure cannot be verified, omit it or mark it
    as unavailable.
12. Do not assume that a reported increase is necessarily
    favourable or unfavourable without context.

==================================================
STRICT REASONING RESTRICTIONS
==================================================

1. Never invent a causal explanation for a financial change.
2. Do not claim a business decision caused a financial
   change unless the source establishes that relationship.
3. Do not classify income as operating or non-operating
   without supporting accounting evidence.
4. Do not equate net worth with regulatory capital adequacy.
5. Do not claim margins compressed without calculating
   and supporting the relevant margins.
6. Do not assume increased expenses represent productive
   investment.
7. Do not assume impairment growth proves widespread
   credit stress.
8. Do not assume missing asset-quality data proves
   poor asset quality.
9. Do not claim a regulatory or corporate restructuring
   caused a financial change without supporting evidence.
10. Do not describe a balance sheet as extraordinarily
    secure without sufficient evidence.
11. Do not give a recommendation based on share-price
    valuation unless verified market valuation data
    has been supplied.
12. Use the currency reported in the source documents.
13. Clearly distinguish reported facts and interpretations.
14. Explain the limitations of relying on a limited
    document set or a single reporting year.
15. Do not treat the absence of evidence as evidence
    that a risk or liability does not exist.

==================================================
OUTPUT FORMAT
==================================================

INVESTMENT COMMITTEE JUDGEMENT

1. EVALUATION OF BULL AGENT

Evaluate:
- Strongest supported arguments.
- Verified evidence.
- Reasonable inferences.
- Unsupported claims and assumptions.

2. EVALUATION OF BEAR AGENT

Evaluate:
- Strongest supported arguments.
- Verified evidence.
- Reasonable inferences.
- Unsupported claims and assumptions.

3. STRONGEST BULLISH ARGUMENT

Explain the strongest evidence-backed bullish argument.

4. STRONGEST BEARISH ARGUMENT

Explain the strongest evidence-backed bearish argument.

5. CRITICAL EVIDENCE COMPARISON

Compare relevant areas supported by the uploaded documents:

- Revenue and income.
- Profitability.
- Income quality.
- Expense growth.
- Assets and liabilities.
- Debt and cash flow.
- Business growth.
- Impairment and financial risks.
- Financial position.

Use separate standalone and consolidated figures
where applicable.

Do not force a comparison when information is unavailable.

6. UNSUPPORTED CLAIMS IDENTIFIED

List important claims from either side that cannot
be established from the supplied documents.

7. UNRESOLVED RISKS AND UNCERTAINTIES

Identify missing information and unresolved disagreements.

8. FINAL RECOMMENDATION

Choose exactly one:

BUY / HOLD / AVOID

9. DECIDING REASONS

Explain why the chosen recommendation is more appropriate
than the other two choices.

10. CONFIDENCE LEVEL

LOW / MEDIUM / HIGH

Explain the limitations affecting confidence.

==================================================
FINAL INSTRUCTION
==================================================

Your decision must be based on evidence, not persuasion.

Do not favour Bull or Bear automatically.

The Risk Agent may apply a separate veto after your decision.
Do not bypass the committee's risk-veto mechanism.

Return a complete, structured judgement.
""",

            tools=[
                semantic_search_annual_report,
                search_annual_report,
            ],
        )

        # -------------------------------------------------
        # STEP 4: PROVIDE DEBATE AND EVIDENCE
        # -------------------------------------------------

        prompt = f"""
Evaluate this completed investment committee debate.

==================================================
SELECTED COMPANY
==================================================

{company}

==================================================
COMPLETE DEBATE TRANSCRIPT
==================================================

{debate}

==================================================
ADDITIONAL SOURCE-DOCUMENT EVIDENCE
==================================================

{verified_evidence}

==================================================
YOUR TASK
==================================================

1. Evaluate the Bull and Bear arguments fairly.
2. Audit important claims against the supplied documents.
3. Separate verified facts, reasonable inferences
   and unsupported claims.
4. Evaluate both rebuttals.
5. Identify the strongest argument from each side.
6. Identify unresolved risks.
7. Produce exactly one final recommendation:
   BUY, HOLD or AVOID.
8. Explain your deciding reasons.
9. State your confidence level.

Do not skip any required output section.
Do not invent evidence or financial figures.
"""

        print("\n[JUDGE] Running Microsoft Agent Framework Agent...")

        response = await judge_agent.run(prompt)

        if not response.text or not response.text.strip():
            raise RuntimeError(
                "Judge Agent returned an empty response."
            )

        print("\n========== JUDGE AGENT RESULT ==========\n")
        print(response.text)

        return response.text


# =====================================================
# MAIN COMMITTEE INTERFACE
# =====================================================

def run_judge_agent():

    print("\n========== JUDGE AGENT ==========")

    print("\n[JUDGE] Running complete debate...")

    debate = run_rebuttal_agent()

    if not debate:
        print("Judge cannot proceed: debate data missing.")
        return None

    try:
        decision = asyncio.run(
            _run_framework_judge(debate)
        )

    except Exception as error:
        print("\n[JUDGE AGENT ERROR]", error)
        raise

    if not decision:
        print("Judge Agent returned no decision.")
        return None

    return decision


# =====================================================
# DIRECT EXECUTION
# =====================================================

if __name__ == "__main__":

    result = run_judge_agent()

    if result:
        print("\nJudge execution completed successfully.")
    else:
        print("\nJudge execution failed.")

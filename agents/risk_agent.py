
import asyncio
import re

from dotenv import load_dotenv
from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient

from agents.rebuttal_agent import run_rebuttal_agent

from tools.research_tools import (
    get_active_company,
)

load_dotenv()


# =====================================================
# RISK WEIGHTS
# =====================================================

RISK_WEIGHTS = {
    "Fair value gains": 20,
    "Declining interest income": 25,
    "Subsidiary impairment": 25,
    "Missing asset quality data": 15,
    "Rising expenses": 15,
}

RISK_THRESHOLD = 70


# =====================================================
# TOOL 1: DETECT RISK INDICATORS
# =====================================================

def identify_risk_indicators(debate: str) -> list:
    """
    Detect predefined risk indicators from the debate.

    Important:
    This is a rule-based screening tool, not a financial
    risk model or a statistically calibrated probability.

    It detects explicit risk-related language. A detected
    indicator still requires source-document verification.
    """

    if not isinstance(debate, str) or not debate.strip():
        raise ValueError("A valid debate transcript is required.")

    text = re.sub(r"\s+", " ", debate.lower())

    patterns = {
        "Fair value gains": [
            r"reliance on fair value gains?",
            r"heavy reliance on fair value",
            r"significant exposure to fair value volatility",
            r"volatile fair value gains?",
            r"fair value gains?.{0,100}(?:volatile|volatility|uncertain|market.sensitive)",
            r"(?:volatile|volatility|uncertain|market.sensitive).{0,100}fair value gains?",
        ],

        "Declining interest income": [
            r"interest income.{0,100}(?:plummet\w*|drop\w*|declin\w*|fall\w*|contract\w*|collaps\w*)",
            r"(?:plummet\w*|drop\w*|declin\w*|fall\w*|contract\w*|collaps\w*).{0,100}interest income",
            r"contraction of core interest income",
            r"decline in core interest generation",
        ],

        "Subsidiary impairment": [
            r"subsidiary.{0,80}(?:impairment charge|impairment loss|impairment indicator)",
            r"(?:impairment charge|impairment loss|impairment indicator).{0,80}subsidiar\w*",
            r"impairment of investments in subsidiaries",
            r"subsidiary investment.{0,100}(?:impaired|impairment)",
            r"subsidiar\w*.{0,100}recoverable amount.{0,100}impairment",
        ],

        "Missing asset quality data": [
            r"(?:asset quality|gnpa|nnpa|non.performing assets|credit risk metrics).{0,100}(?:not disclosed|not available|not provided|insufficient disclosure|unavailable)",
            r"(?:not disclosed|not available|not provided|insufficient disclosure|unavailable).{0,100}(?:asset quality|gnpa|nnpa|non.performing assets|credit risk metrics)",
            r"absence of (?:granular )?asset quality disclosure",
            r"lack of (?:granular )?asset quality information",
        ],

        "Rising expenses": [
            r"(?:total|operating|employee|staff|administrative) expenses?.{0,100}(?:increas\w*|ris\w*|doubl\w*|jump\w*|escalat\w*)",
            r"(?:increas\w*|ris\w*|doubl\w*|jump\w*|escalat\w*).{0,100}(?:total|operating|employee|staff|administrative) expenses?",
            r"expenses? increased significantly",
            r"expense growth outpaced revenue growth",
        ],
    }

    indicators = []
    matched_evidence = {}

    for risk, expressions in patterns.items():

        matches = []

        for expression in expressions:
            found = re.search(expression, text)

            if found:
                matches.append(found.group(0))

        if matches:
            indicators.append(risk)
            matched_evidence[risk] = list(dict.fromkeys(matches))

    print("\n========== RISK DETECTION RESULTS ==========")

    for risk in RISK_WEIGHTS:

        if risk in matched_evidence:
            print(f"\n{risk}: DETECTED")

            for evidence in matched_evidence[risk][:3]:
                print("Evidence:", evidence)

        else:
            print(f"\n{risk}: NOT DETECTED")

    print(
        "\nNOTE: Detection is based on debate text. "
        "It does not independently prove that a financial risk exists."
    )

    return indicators


# =====================================================
# TOOL 2: CALCULATE RISK SCORE
# =====================================================

def calculate_risk_score(indicators: list) -> int:
    """
    Calculate the weighted screening score.

    Maximum score: 100.
    Each category is counted once.
    """

    if not isinstance(indicators, (list, tuple, set)):
        raise TypeError("Indicators must be a list, tuple or set.")

    unknown = set(indicators) - set(RISK_WEIGHTS)

    if unknown:
        raise ValueError(
            f"Unknown risk indicators: {sorted(unknown)}"
        )

    score = sum(
        RISK_WEIGHTS[indicator]
        for indicator in set(indicators)
    )

    return min(score, 100)


# =====================================================
# MICROSOFT AGENT FRAMEWORK RISK AGENT
# =====================================================

async def _run_framework_risk_agent(
    company: str,
    debate: str,
    indicators: list,
    risk_score: int,
    veto: bool,
):

    print("\n========== MICROSOFT FRAMEWORK RISK AGENT ==========")
    print(f"[RISK AGENT] Selected company: {company}")

    # -------------------------------------------------
    # STEP 1: PREPARE RISK INFORMATION
    # -------------------------------------------------

    risk_details = []

    for indicator in indicators:
        weight = RISK_WEIGHTS.get(indicator, 0)

        risk_details.append(
            f"{indicator}: {weight} points"
        )

    risk_details_text = (
        "\n".join(risk_details)
        if risk_details
        else "No predefined indicators detected."
    )

    # -------------------------------------------------
    # STEP 2: CREATE FRAMEWORK AGENT
    # -------------------------------------------------

    async with GeminiChatClient(
        model="gemini-3.5-flash-lite"
    ) as client:

        risk_agent = Agent(
            client=client,
            name="RiskAgent",

            instructions="""
You are the RISK AGENT of a Debate-Driven Investment Committee.

You independently explain financial risks identified in the
Research, Bull, Bear and Rebuttal stages.

You are implemented using Microsoft Agent Framework.

==================================================
AVAILABLE PYTHON TOOLS
==================================================

TOOL 1: identify_risk_indicators(debate)

Detects predefined risk-related phrases in the debate.

TOOL 2: calculate_risk_score(indicators)

Calculates a weighted screening score.

Both tools have already been executed by the workflow.
Their results are supplied in the input.

You may call these tools again if necessary.

==================================================
RISK SCORING SYSTEM
==================================================

Fair value gains: 20 points

Declining interest income: 25 points

Subsidiary impairment: 25 points

Missing asset quality data: 15 points

Rising expenses: 15 points

Maximum risk score: 100.

Risk threshold: 70.

A score equal to or above 70 triggers a veto.

==================================================
RISK ASSESSMENT RULES
==================================================

1. Explain every detected indicator using the debate evidence.

2. Distinguish reported financial facts from interpretations.

3. Do not invent financial figures or facts.

4. Do not claim an impairment indicator means an actual
   impairment charge or write-off.

5. Missing asset quality information is uncertainty,
   not proof of poor performance.

6. Do not invent reasons for declining interest income.

7. Do not automatically classify fair value gains
   as non-operating income.

8. Do not assume rising expenses prove inefficiency.

9. Do not treat a percentage increase in impairment
   as proof of widespread credit stress.

10. Do not equate net worth with regulatory capital adequacy.

11. Do not treat AUM, app downloads or transaction volume
    as revenue.

12. Explain the limitations of keyword-based risk detection.

13. The weights and threshold are fixed. Do not change them.

14. State the veto decision accurately.

15. The Risk Agent operates independently of the Judge's
    recommendation.

16. Assess only the selected company.

17. Do not treat a detected phrase as verified financial evidence.

18. Some risk categories may not apply to every industry.
    State when a category is not meaningfully applicable.

==================================================
IMPORTANT VETO RULE
==================================================

If risk score >= 70:

VETO = TRUE.

The committee controller must override a BUY recommendation
with HOLD.

If risk score < 70:

VETO = FALSE.

The Judge's recommendation remains subject to the
committee controller's other decision rules.

The Risk Agent does not independently choose BUY, HOLD or AVOID.

==================================================
SCORING LIMITATIONS
==================================================

This score is a predefined academic screening heuristic.

It is NOT:
- A probability of financial failure.
- A statistically calibrated risk model.
- A credit rating.
- A substitute for financial statement analysis.

The score depends on the wording of the debate transcript.
It may miss risks that are not mentioned and may detect
risks that need further verification.

==================================================
OUTPUT FORMAT
==================================================

RISK AGENT ASSESSMENT

1. SELECTED COMPANY

2. DETECTED RISK INDICATORS

3. EXPLANATION OF EACH RISK

4. POTENTIAL FINANCIAL IMPACT

5. RISK SCORE ANALYSIS

6. LIMITATIONS OF RISK DETECTION AND SCORING

7. FINAL VETO DECISION

Clearly state:
- Risk score.
- Risk threshold.
- Whether veto is triggered.
- Why the veto rule applies.

This is an academic project, not financial advice.

Return a complete and structured assessment.
""",

            tools=[
                identify_risk_indicators,
                calculate_risk_score,
            ],
        )

        # -------------------------------------------------
        # STEP 3: PROVIDE RISK INFORMATION
        # -------------------------------------------------

        prompt = f"""
Perform an independent risk assessment.

==================================================
SELECTED COMPANY
==================================================

{company}

==================================================
COMPLETE DEBATE TRANSCRIPT
==================================================

{debate}

==================================================
DETECTED RISK INDICATORS
==================================================

{risk_details_text}

==================================================
RISK SCORE
==================================================

{risk_score}/100

==================================================
RISK THRESHOLD
==================================================

{RISK_THRESHOLD}/100

==================================================
VETO STATUS
==================================================

{veto}

==================================================
YOUR TASK
==================================================

1. Explain each detected risk.
2. Distinguish facts from interpretations.
3. Explain potential financial impact.
4. Evaluate the limitations of keyword-based detection.
5. Explain the calculated score.
6. State whether the veto is triggered.
7. Identify any detected indicator that needs further
   verification against the original company documents.
8. Explain whether any category may not apply to this
   company's industry.

Do not change the calculated risk score.
Do not invent financial evidence.
Do not independently choose BUY, HOLD or AVOID.
"""

        print("\n[RISK AGENT] Running Microsoft Agent Framework...")

        response = await risk_agent.run(prompt)

        if not response.text or not response.text.strip():
            raise RuntimeError(
                "Risk Agent returned an empty response."
            )

        print("\n========== FRAMEWORK RISK REPORT ==========\n")
        print(response.text)

        return response.text


# =====================================================
# RISK AGENT EXECUTION
# =====================================================

def run_risk_agent(debate=None):

    print("\n========== RISK AGENT ==========")

    company = get_active_company()

    print(f"[RISK AGENT] Assessing: {company}")

    # -------------------------------------------------
    # STEP 1: OBTAIN DEBATE TRANSCRIPT
    # -------------------------------------------------

    if debate is None:

        print("\nGenerating debate transcript...")

        debate = run_rebuttal_agent(company)

    if not debate:
        print("\nRisk assessment failed: debate unavailable.")
        return None

    # -------------------------------------------------
    # STEP 2: EXECUTE TOOL 1
    # -------------------------------------------------

    print("\n[RISK TOOL 1/2] Detecting risk indicators...")

    indicators = identify_risk_indicators(debate)

    print("\n[RISK AGENT] Risk detection completed.")

    # -------------------------------------------------
    # STEP 3: EXECUTE TOOL 2
    # -------------------------------------------------

    print("\n[RISK TOOL 2/2] Calculating weighted risk score...")

    risk_score = calculate_risk_score(indicators)

    print("[RISK AGENT] Risk score calculated.")

    # -------------------------------------------------
    # STEP 4: APPLY VETO THRESHOLD
    # -------------------------------------------------

    veto = risk_score >= RISK_THRESHOLD

    print("\n========== RISK SCORE ==========")

    print("Company:", company)
    print("Risk Score:", risk_score, "/100")
    print("Risk Threshold:", RISK_THRESHOLD, "/100")
    print("Veto Triggered:", veto)

    # -------------------------------------------------
    # STEP 5: RUN FRAMEWORK RISK AGENT
    # -------------------------------------------------

    try:
        explanation = asyncio.run(
            _run_framework_risk_agent(
                company,
                debate,
                indicators,
                risk_score,
                veto,
            )
        )

    except Exception as error:
        print("\n[RISK AGENT ERROR]", error)
        raise

    # -------------------------------------------------
    # STEP 6: DISPLAY FINAL ASSESSMENT
    # -------------------------------------------------

    print("\n========== FINAL RISK ASSESSMENT ==========")

    print("\nDetected Risks:")

    if indicators:
        for indicator in indicators:
            print("-", indicator)
    else:
        print("No predefined risk indicators detected.")

    print("\nRisk Score:", risk_score, "/100")
    print("Risk Threshold:", RISK_THRESHOLD, "/100")
    print("Veto Triggered:", veto)

    print("\n========== RISK AGENT COMPLETED ==========")

    # -------------------------------------------------
    # STEP 7: RETURN CONTROLLER-COMPATIBLE RESULT
    # -------------------------------------------------

    return {
        "company": company,
        "risk_indicators": indicators,
        "risk_score": risk_score,
        "threshold": RISK_THRESHOLD,
        "veto": veto,
        "explanation": explanation,
    }


# =====================================================
# STANDALONE EXECUTION
# =====================================================

if __name__ == "__main__":

    result = run_risk_agent()

    if result:
        print("\n========== RISK AGENT COMPLETED ==========")
        print("Company:", result["company"])
        print("Final Score:", result["risk_score"])
        print("Veto:", result["veto"])
    else:
        print("\nRisk Agent execution failed.")

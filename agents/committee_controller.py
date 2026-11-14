
import asyncio
import re

from agents.rebuttal_agent import run_rebuttal_agent
from agents.judge_agent import _run_framework_judge
from agents.risk_agent import run_risk_agent
from logs.run_logger import log_committee_run


# =====================================================
# TOOL 1: EXTRACT JUDGE RECOMMENDATION
# =====================================================

def extract_recommendation(judge_output):

    if not judge_output:
        return "HOLD"

    # Prefer an explicitly labelled final recommendation.
    patterns = [
        r"FINAL\s+RECOMMENDATION\s*:?\s*[\*_`#\s]*(BUY|HOLD|AVOID)\b",
        r"FINAL\s+INVESTMENT\s+RECOMMENDATION\s*:?\s*[\*_`#\s]*(BUY|HOLD|AVOID)\b",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            judge_output,
            re.IGNORECASE
        )

        if matches:
            return matches[-1].upper()

    # Fallback: look for the recommendation section.
    section = re.search(
        r"8\.\s*FINAL RECOMMENDATION(.*?)(?=\n\s*9\.|\Z)",
        judge_output,
        re.IGNORECASE | re.DOTALL
    )

    if section:

        match = re.search(
            r"\b(BUY|HOLD|AVOID)\b",
            section.group(1),
            re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

    # Never guess if the Judge's decision is ambiguous.
    return "HOLD"


# =====================================================
# TOOL 2: APPLY RISK VETO
# =====================================================

def apply_risk_veto(judge_recommendation, risk_result):

    # Risk assessment failure: fail-safe HOLD.
    if not risk_result:

        return (
            "HOLD",
            "Risk assessment unavailable. Defaulting to HOLD."
        )

    if "veto" not in risk_result:

        return (
            "HOLD",
            "Risk veto status unavailable. Defaulting to HOLD."
        )

    if judge_recommendation not in ["BUY", "HOLD", "AVOID"]:

        return (
            "HOLD",
            "Invalid Judge recommendation. Defaulting to HOLD."
        )

    # Validate the veto value.
    if not isinstance(risk_result["veto"], bool):

        return (
            "HOLD",
            "Invalid Risk Agent veto status. Defaulting to HOLD."
        )

    # Veto overrides BUY only.
    if risk_result["veto"] and judge_recommendation == "BUY":

        return (
            "HOLD",
            "Risk Agent veto triggered. BUY overridden to HOLD."
        )

    # Judge already recommends HOLD.
    if risk_result["veto"] and judge_recommendation == "HOLD":

        return (
            "HOLD",
            "Judge recommended HOLD. Risk veto independently confirms the cautious decision."
        )

    # Risk veto does not override AVOID.
    if risk_result["veto"] and judge_recommendation == "AVOID":

        return (
            "AVOID",
            "Judge recommended AVOID. Risk veto does not override AVOID."
        )

    return (
        judge_recommendation,
        "No risk veto triggered. Judge decision retained."
    )


# =====================================================
# JUDGE AGENT
# =====================================================

def run_committee_judge(debate):

    print("\n========== JUDGE AGENT ==========")

    if not debate:

        return {
            "recommendation": "HOLD",
            "analysis": "Judge cannot proceed: debate unavailable."
        }

    try:

        # Reuse the existing debate transcript.
        # Do not generate another Research/Bull/Bear/Rebuttal run.

        judge_result = asyncio.run(
            _run_framework_judge(debate)
        )

        print("\n========== JUDGE RESULT ==========")
        print(judge_result)

        recommendation = extract_recommendation(judge_result)

        print(
            "\nExtracted Judge Recommendation:",
            recommendation
        )

        return {
            "recommendation": recommendation,
            "analysis": judge_result
        }

    except Exception as error:

        print("\nJudge Agent error:", error)

        return {
            "recommendation": "HOLD",
            "analysis": f"Judge assessment failed: {error}"
        }


# =====================================================
# MAIN COMMITTEE CONTROLLER
# =====================================================

def run_committee(company_name="Jio Financial Services"):

    print("\n" + "=" * 60)
    print("       DEBATE-DRIVEN INVESTMENT COMMITTEE")
    print("=" * 60)

    print("\nSelected Company:", company_name)

    # -------------------------------------------------
    # STEP 1: GENERATE DEBATE
    # -------------------------------------------------

    print("\nSTEP 1: RUNNING RESEARCH, BULL, BEAR AND REBUTTAL")

    try:

        # Pass the selected company into the entire debate.
        debate = run_rebuttal_agent(company_name)

    except Exception as error:

        print("\nDebate execution failed:", error)

        return {
            "company": company_name,
            "judge_recommendation": "HOLD",
            "judge_analysis": "Debate execution failed.",
            "risk_result": None,
            "final_recommendation": "HOLD",
            "decision_reason": "Debate unavailable. Defaulting to HOLD."
        }

    if not debate:

        print("\nDebate unavailable. Defaulting to HOLD.")

        return {
            "company": company_name,
            "judge_recommendation": "HOLD",
            "judge_analysis": "Debate transcript unavailable.",
            "risk_result": None,
            "final_recommendation": "HOLD",
            "decision_reason": "Debate unavailable. Defaulting to HOLD."
        }

    # -------------------------------------------------
    # STEP 2: RUN JUDGE AGENT
    # -------------------------------------------------

    print("\nSTEP 2: RUNNING MICROSOFT FRAMEWORK JUDGE")

    judge_result = run_committee_judge(debate)

    judge_recommendation = judge_result["recommendation"]

    print("\nJudge Recommendation:", judge_recommendation)

    # -------------------------------------------------
    # STEP 3: RUN RISK AGENT
    # -------------------------------------------------

    print("\nSTEP 3: RUNNING MICROSOFT FRAMEWORK RISK AGENT")

    try:

        # Pass the same debate transcript.
        # Do not generate another debate.

        risk_result = run_risk_agent(debate)

    except Exception as error:

        print("\nRisk Agent execution failed:", error)

        risk_result = None

    # -------------------------------------------------
    # STEP 4: APPLY RISK VETO
    # -------------------------------------------------

    print("\nSTEP 4: APPLYING RISK VETO")

    final_recommendation, decision_reason = apply_risk_veto(
        judge_recommendation,
        risk_result
    )

    # -------------------------------------------------
    # STEP 5: FINAL COMMITTEE DECISION
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("       FINAL INVESTMENT COMMITTEE DECISION")
    print("=" * 60)

    print("\nCompany:", company_name)

    print("Judge Recommendation:", judge_recommendation)

    if risk_result:

        print(
            "Risk Score:",
            risk_result.get("risk_score", "Unavailable"),
            "/100"
        )

        print(
            "Risk Threshold:",
            risk_result.get("threshold", "Unavailable"),
            "/100"
        )

        print(
            "Risk Veto:",
            risk_result.get("veto", "Unavailable")
        )

    else:

        print("Risk Score: Unavailable")
        print("Risk Veto: Unavailable")

    print("\nFINAL RECOMMENDATION:", final_recommendation)

    print("Decision Reason:", decision_reason)

    # -------------------------------------------------
    # STEP 6: SAVE COMMITTEE EXECUTION
    # -------------------------------------------------

    print("\nSTEP 6: SAVING COMMITTEE RUN")

    try:

        log_committee_run(
            debate=debate,
            judge_result=judge_result,
            risk_result=risk_result,
            final_recommendation=final_recommendation,
            decision_reason=decision_reason,
            company=company_name
        )

        print("[LOGGER] Committee execution saved.")

    except Exception as error:

        # Logging failure must not change the decision.

        print(
            "[LOGGER WARNING] Could not save committee run:",
            error
        )

    # -------------------------------------------------
    # STEP 7: RETURN FINAL RESULT
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("       COMMITTEE PROCESS COMPLETED")
    print("=" * 60)

    return {
        "company": company_name,
        "judge_recommendation": judge_recommendation,
        "judge_analysis": judge_result["analysis"],
        "risk_result": risk_result,
        "final_recommendation": final_recommendation,
        "decision_reason": decision_reason,
        "debate": debate
    }


# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":

    result = run_committee()

    print("\n========== FINAL RESULT OBJECT ==========")

    if result:

        print("Company:", result["company"])
        print("Judge:", result["judge_recommendation"])
        print("Final:", result["final_recommendation"])
        print("Reason:", result["decision_reason"])

    else:

        print("Committee execution failed.")

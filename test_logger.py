
from logs.run_logger import (
    log_committee_run,
    read_committee_logs
)

# Sample execution data for testing only
debate = {
    "bull": "Potential business growth",
    "bear": "Fair value gains and declining interest income",
    "rebuttal": "Both sides responded to opposing arguments"
}

judge_result = {
    "recommendation": "HOLD",
    "confidence": "Medium"
}

risk_result = {
    "risk_score": 85,
    "threshold": 70,
    "veto": True
}

log_committee_run(
    debate=debate,
    judge_result=judge_result,
    risk_result=risk_result,
    final_recommendation="HOLD",
    decision_reason="Risk Agent veto triggered"
)

# Verify saved records
records = read_committee_logs()

print("\nTotal saved runs:", len(records))
print("Latest recommendation:",
      records[-1]["final_recommendation"])

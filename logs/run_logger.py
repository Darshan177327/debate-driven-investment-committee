
import json
from datetime import datetime, timezone
from pathlib import Path


# Store logs in the project's logs folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "committee_runs.jsonl"


def log_committee_run(
    debate,
    judge_result,
    risk_result,
    final_recommendation,
    decision_reason,
    company="Jio Financial Services"
):
    """
    Append one committee execution to the run log.
    Existing records are never overwritten.
    """

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "asset": company,

        "debate": debate,

        "judge_result": judge_result,

        "risk_result": risk_result,

        "final_recommendation": final_recommendation,

        "decision_reason": decision_reason
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str))
        file.write("\n")

    print(f"\nRun successfully logged: {LOG_FILE}")


def read_committee_logs():
    """Read all previously recorded committee executions."""

    if not LOG_FILE.exists():
        return []

    records = []

    with LOG_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records

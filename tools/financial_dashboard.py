"""Conservative first-pass parser for financial statement line items.

It intentionally emits only rows where a recognised metric and at least two
financial-year labels/numbers occur on the same extracted text line.
"""
from __future__ import annotations

import re

METRIC_PATTERNS = {
    "Revenue from operations": r"revenue\s+from\s+operations",
    "Total income": r"total\s+income",
    "Interest income": r"interest\s+income",
    "Dividend income": r"dividend\s+income",
    "Fees and commission": r"fees?\s+and\s+commission",
    "Fair value gains": r"(?:net\s+)?gain[s]?\s+(?:on\s+)?fair\s+value|fair\s+value\s+(?:gain|changes?)",
    "Total expenses": r"total\s+expenses?",
    "Profit before tax": r"profit\s+before\s+tax",
    "Profit after tax": r"profit\s+(?:for\s+the\s+year|after\s+tax)|net\s+profit",
    "Earnings per share": r"(?:basic|diluted)?\s*earnings?\s+per\s+share|eps\b",
    "Impairment expense": r"impairment\s+(?:loss|charge|on\s+financial\s+instruments)",
    "Borrowings": r"borrowings",
}

YEAR_RE = re.compile(r"\b(?:FY\s*)?(20\d{2})\s*[-/]\s*(?:FY\s*)?(?:20)?(\d{2})\b|\bFY\s*(20\d{2})\b", re.I)
NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?\d[\d,]*(?:\.\d+)?\)?")

def _periods(line: str) -> list[str]:
    found = []
    for match in YEAR_RE.finditer(line):
        if match.group(3):
            year = int(match.group(3))
            label = f"FY{year}"
        else:
            start = int(match.group(1))
            end = int(match.group(2))
            label = f"FY{start}-{end:02d}"
        if label not in found:
            found.append(label)
    return found

def _to_number(value: str):
    value = value.replace(",", "").strip()
    negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()")
    try:
        number = float(value)
        return -number if negative else number
    except ValueError:
        return None

def extract_financial_metrics(text: str, source: str) -> list[dict]:
    output = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or len(line) > 500:
            continue
        periods = _periods(line)
        if len(periods) < 2:
            continue
        metric = next((name for name, pattern in METRIC_PATTERNS.items()
                       if re.search(pattern, line, re.I)), None)
        if not metric:
            continue
        # Remove the metric label and year labels before extracting values.
        remainder = line
        for pattern in METRIC_PATTERNS.values():
            remainder = re.sub(pattern, " ", remainder, flags=re.I)
        remainder = YEAR_RE.sub(" ", remainder)
        values = [_to_number(x) for x in NUMBER_RE.findall(remainder)]
        values = [x for x in values if x is not None]
        if len(values) < 2:
            continue
        # Use the first two values only when the line has exactly two periods.
        # If extra numeric columns are present, reject rather than guess.
        if len(values) != len(periods):
            continue
        for period, value in zip(periods, values):
            output.append({
                "metric": metric,
                "period": period,
                "value": value,
                "source": source,
                "evidence_line": line,
            })
    return output

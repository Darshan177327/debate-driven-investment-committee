"""Conservative financial metric extraction from company annual-report chunks.

Supports (1) financial highlights written as prose/cards, and (2) comparative
financial tables where labels, years and values occur in one indexed chunk.
Only figures with explicit financial context are emitted. Source evidence and
scope/unit metadata are retained for verification.
"""
from __future__ import annotations

import re
from typing import Iterable

METRIC_PATTERNS = {
    "Revenue from operations": r"revenue\s+from\s+operations|revenue\s+from\s+contracts?\s+with\s+customers",
    "Total revenue": r"(?<!from\s)\btotal\s+revenue\b|\bturnover\b|(?<!adjusted )\brevenue\b(?!\s+from)",
    "Income from business operations": r"\bincome\s+from\s+(?:core\s+)?business\s+operations\b",
    "Adjusted revenue": r"\badjusted\s+revenue\b",
    "Total income": r"\btotal\s+income\b",
    "Other income": r"\bother\s+income\b",
    "Interest income": r"\binterest\s+income\b",
    "Dividend income": r"\bdividend\s+income\b",
    "Fees and commission income": r"fees?\s+and\s+commission\s+(?:income|revenue)",
    "Total expenses": r"\btotal\s+expenses?\b",
    "Employee expenses": r"employee\s+benefit\s+expenses?|employee\s+expenses?",
    "Finance costs": r"\bfinance\s+costs?\b|\binterest\s+expenses?\b",
    "Depreciation and amortisation": r"depreciation\s+and\s+amortisation|depreciation\s+and\s+amortization",
    "Profit before tax": r"\bprofit\s+before\s+tax(?:ation)?\b|\bprofit\s+before\s+exceptional\s+items\s+and\s+tax\b",
    "Profit after tax": r"\bprofit\s+after\s+tax\b|\bprofit\s+for\s+the\s+year\b|\bprofit\s+for\s+the\s+period\b|\bnet\s+profit\b|\bPAT\b",
    "Adjusted EBITDA": r"\badjusted\s+ebitda\b",
    "EBITDA": r"(?<!adjusted\s)\bebitda\b",
    "Total assets": r"\btotal\s+assets\b",
    "Total equity": r"\btotal\s+equity\b|\btotal\s+shareholders?\s+funds\b",
    "Total liabilities": r"\btotal\s+liabilities\b",
    "Borrowings": r"\btotal\s+borrowings\b|\bborrowings\b",
    "Cash and cash equivalents": r"\bcash\s+and\s+cash\s+equivalents?\b",
    "Operating cash flow": r"\bnet\s+cash\s+(?:generated\s+from|from)\s+operating\s+activities\b|\bcash\s+flow\s+from\s+operating\s+activities\b",
    "Investing cash flow": r"\bnet\s+cash\s+(?:used\s+in|from)\s+investing\s+activities\b|\bcash\s+flow\s+from\s+investing\s+activities\b",
    "Financing cash flow": r"\bnet\s+cash\s+(?:used\s+in|from)\s+financing\s+activities\b|\bcash\s+flow\s+from\s+financing\s+activities\b",
    "Earnings per share": r"\b(?:basic|diluted)?\s*earnings?\s+per\s+share\b|\bEPS\b",
}

# Do not match plain years outside explicit FY/year-range contexts.
PERIOD_RE = re.compile(
    r"\b(?:FY\s*)?(20\d{2})\s*[-/]\s*(?:FY\s*)?(20)?(\d{2})\b"
    r"|\bFY\s*(20\d{2})\b|\bFY\s*(\d{2})\b", re.I
)
# A number must be a standalone token; percentages and year components are filtered.
NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?\d[\d,]*(?:\.\d+)?\)?")


def _normalise_period(start: str, century: str | None = None, end: str | None = None) -> str:
    year = int(start)
    if end is not None:
        end_year = int((century or "20") + end) if len(end) == 2 else int(end)
        return f"FY{year}-{end_year % 100:02d}"
    return f"FY{year}"


def _periods(text: str) -> list[tuple[str, tuple[int, int]]]:
    found: list[tuple[str, tuple[int, int]]] = []
    for m in PERIOD_RE.finditer(text):
        if m.group(1):
            label = _normalise_period(m.group(1), m.group(2), m.group(3))
        elif m.group(4):
            label = _normalise_period(m.group(4))
        else:
            label = _normalise_period(str(2000 + int(m.group(5))))
        if all(p != label for p, _ in found):
            found.append((label, m.span()))
    return found


def _filename_period(filename: str) -> str | None:
    # Annual-report filename cues are used only when there is exactly one clear FY.
    matches = re.findall(r"(?:FY\s*)?(20\d{2})\s*[-_]\s*(20)?(\d{2})", filename, re.I)
    if len(matches) == 1:
        a, b, c = matches[0]
        return _normalise_period(a, b, c)
    short = re.findall(r"(?:FY|financial.year)[_ -]?(\d{2})\b", filename, re.I)
    if len(set(short)) == 1:
        return f"FY{2000 + int(short[0]):04d}"
    return None


def _to_number(token: str) -> float | None:
    raw = token.replace(",", "").strip()
    negative = raw.startswith("(") and raw.endswith(")")
    try:
        value = float(raw.strip("()"))
        return -value if negative else value
    except ValueError:
        return None


def _number_matches(text: str) -> list[re.Match[str]]:
    period_spans = [span for _, span in _periods(text)]
    out = []
    for m in NUMBER_RE.finditer(text):
        if any(m.start() < b and m.end() > a for a, b in period_spans):
            continue
        # Exclude percentages, dates and values explicitly denominated in USD.
        after = text[m.end():m.end() + 8]
        before = text[max(0, m.start() - 8):m.start()]
        if re.match(r"\s*%", after) or re.search(r"\b(?:USD|US\$)\s*$", before, re.I):
            continue
        out.append(m)
    return out


def _scope(text: str, start: int, end: int) -> str:
    context = text[max(0, start - 180):min(len(text), end + 100)]
    # Explicit nearby scope only; otherwise retain as Unspecified.
    if re.search(r"\bconsolidated\b", context, re.I):
        return "Consolidated"
    if re.search(r"\bstandalone\b", context, re.I):
        return "Standalone"
    return "Unspecified"


def _unit(text: str, start: int, end: int) -> tuple[str, str]:
    context = text[max(0, start - 100):min(len(text), end + 100)]
    if re.search(r"₹|INR|Rs\.?", context, re.I):
        currency = "INR"
    elif re.search(r"US\$|\bUSD\b", context, re.I):
        currency = "USD"
    else:
        currency = "Unspecified"
    if re.search(r"\b(?:crore|crores)\b|₹\s*crore|INR\s*crore", context, re.I):
        unit = "crore"
    elif re.search(r"\b(?:million|mn)\b", context, re.I):
        unit = "million"
    elif re.search(r"\b(?:billion|bn)\b", context, re.I):
        unit = "billion"
    elif re.search(r"\bper\s+share\b|\bEPS\b", context, re.I):
        unit = "per_share"
    else:
        unit = "Unspecified"
    return currency, unit


def _make_row(metric: str, period: str, value: float, filename: str,
              chunk_index: object, evidence: str, scope: str, currency: str,
              unit: str, extraction: str) -> dict:
    return {
        "metric": metric, "period": period, "value": value,
        "source": filename, "chunk": chunk_index, "evidence": evidence,
        "scope": scope, "currency": currency, "unit": unit,
        "extraction_method": extraction,
    }


def _extract_single_pass(chunks: Iterable[dict]) -> list[dict]:
    """Extract metric-period pairs from each supplied text passage."""
    rows: list[dict] = []
    seen: set[tuple] = set()
    compiled = [(name, re.compile(pattern, re.I)) for name, pattern in METRIC_PATTERNS.items()]

    for item in chunks:
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        meta = item.get("metadata") or {}
        filename = str(meta.get("filename") or "Indexed document")
        chunk_index = meta.get("chunk_index")
        if not text:
            continue
        periods = _periods(text)
        fallback_period = _filename_period(filename)
        matches = []
        for metric, pattern in compiled:
            for m in pattern.finditer(text):
                matches.append((m.start(), m.end(), metric))
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        filtered = []
        for cur in matches:
            if filtered and cur[0] < filtered[-1][1]:
                continue
            filtered.append(cur)

        for i, (start, end, metric) in enumerate(filtered):
            prev_end = filtered[i-1][1] if i else max(0, start-100)
            next_start = filtered[i+1][0] if i+1 < len(filtered) else min(len(text), end+100)
            # The nearest metric value may occur immediately before or after
            # its label in designed financial-highlight cards.
            left = max(0, start-38)
            right = min(len(text), end+38)
            local = text[left:right]
            local_periods = _periods(local)
            period = local_periods[0][0] if len(local_periods) == 1 else None
            if period is None and len(periods) == 1:
                period = periods[0][0]
            if period is None and not periods:
                period = fallback_period

            # First handle explicit comparative rows. A row is accepted only
            # when the number of periods and values match exactly.
            if len(periods) >= 2:
                row_text = text[end:next_start]
                row_nums = _number_matches(row_text)
                header = text[max(0, start-650):start]
                # Reject mixed INR/USD tables unless INR column structure is
                # unambiguous; never guess which interleaved column is INR.
                mixed_currency = bool(re.search(r"US\$|\bUSD\b", header, re.I)) and bool(re.search(r"₹|\bINR\b", header, re.I))
                if not mixed_currency and len(row_nums) == len(periods):
                    for (p, _), nm in zip(periods, row_nums):
                        value = _to_number(nm.group())
                        if value is None:
                            continue
                        cur, unit = _unit(text, end+nm.start(), end+nm.end())
                        ev = text[max(0, start-100):min(len(text), next_start+80)]
                        row = _make_row(metric, p, value, filename, chunk_index, ev,
                                        _scope(text, start, next_start), cur, unit, "comparative_table")
                        key = (metric, p, value, filename, chunk_index)
                        if key not in seen:
                            seen.add(key); rows.append(row)
                    continue

            if not period:
                continue
            # Find explicit financial amounts close to the label, on either
            # side. Ignore percentages, growth figures, years and share counts.
            candidates = []
            for nm in _number_matches(local):
                token = local[max(0, nm.start()-12):min(len(local), nm.end()+12)]
                if re.search(r"\b(?:YoY|growth|improvement|increase|decrease)\b", token, re.I):
                    continue
                if not re.search(r"₹|\bINR\b|\bRs\\.?\b|\bcrores?\b|\bper\s+share\b", token, re.I):
                    continue
                # Reject values attached to percentage notation.
                if re.search(r"%", token):
                    continue
                # Do not cross another financial metric label: this prevents
                # EBITDA's amount being reused for PAT, or EPS for income.
                abs_num_mid = left + (nm.start()+nm.end())/2
                between = text[int(min(abs_num_mid, (start+end)/2)):int(max(abs_num_mid, (start+end)/2))]
                other_metric_between = any(
                    other_name != metric and other_pattern.search(between)
                    for other_name, other_pattern in compiled
                )
                if other_metric_between:
                    continue
                candidates.append(nm)
            if not candidates:
                continue
            label_mid = (start + end) / 2 - left
            before_label = [n for n in candidates if (n.start()+n.end())/2 <= label_mid]
            nm = max(before_label, key=lambda n: n.end()) if before_label else min(candidates, key=lambda n: abs((n.start()+n.end())/2-label_mid))
            value = _to_number(nm.group())
            if value is None:
                continue
            abs_start, abs_end = left+nm.start(), left+nm.end()
            cur, unit = _unit(text, abs_start, abs_end)
            evidence = text[max(0, min(start, abs_start)-70):min(len(text), max(end, abs_end)+70)]
            row = _make_row(metric, period, value, filename, chunk_index, evidence,
                            _scope(text, min(start, abs_start), max(end, abs_end)),
                            cur, unit, "financial_highlight")
            key = (metric, period, value, filename, chunk_index)
            if key not in seen:
                seen.add(key); rows.append(row)
    return rows


def extract_financial_metrics(text: str, source: str) -> list[dict]:
    """Backward-compatible adapter for direct text extraction."""
    return extract_financial_metrics_from_chunks([
        {"text": text, "metadata": {"filename": source, "chunk_index": None}}
    ])


def extract_financial_metrics_from_chunks(chunks: Iterable[dict]) -> list[dict]:
    """Extract from original passages and short adjacent windows.

    Annual-report table headers, row labels and values are frequently split
    between neighbouring ChromaDB chunks. Windows are restricted to the same
    source file and consecutive chunk indexes; the vector store is read-only.
    """
    original = list(chunks)
    candidates = list(original)
    by_file: dict[str, list[dict]] = {}
    for item in original:
        meta = item.get("metadata") or {}
        filename = str(meta.get("filename") or "Indexed document")
        by_file.setdefault(filename, []).append(item)

    for filename, items in by_file.items():
        items.sort(key=lambda x: int((x.get("metadata") or {}).get("chunk_index", 0)))
        for width in (2, 3):
            for start in range(0, len(items) - width + 1):
                window = items[start:start + width]
                indexes = [int((x.get("metadata") or {}).get("chunk_index", 0)) for x in window]
                if indexes != list(range(indexes[0], indexes[0] + width)):
                    continue
                joined = " ".join(str(x.get("text") or "") for x in window)
                if not joined.strip():
                    continue
                meta = dict(window[0].get("metadata") or {})
                meta["chunk_index"] = f"{indexes[0]}-{indexes[-1]}"
                candidates.append({"text": joined, "metadata": meta})

    extracted = _extract_single_pass(candidates)
    # Prefer a single-chunk source where possible; otherwise retain the
    # adjacent-window evidence that made the pairing possible.
    unique: dict[tuple, dict] = {}
    for row in extracted:
        key = (row.get("metric"), row.get("period"), row.get("value"),
               row.get("source"), row.get("scope"), row.get("currency"), row.get("unit"))
        previous = unique.get(key)
        if previous is None or ("-" in str(previous.get("chunk")) and "-" not in str(row.get("chunk"))):
            unique[key] = row
    return list(unique.values())

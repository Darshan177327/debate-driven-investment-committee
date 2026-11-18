from tools.financial_dashboard import extract_financial_metrics_from_chunks


def rows(text, filename, chunk=1):
    return extract_financial_metrics_from_chunks([
        {"text": text, "metadata": {"filename": filename, "chunk_index": chunk}}
    ])


def test_jio_fy25_financial_highlights():
    text = ("Performance Snapshot ₹2,079 crore CONSOLIDATED TOTAL INCOME l "
            "₹1,594 crore PRE-PROVISION OPERATING PROFIT (PPOP) l "
            "₹2.54 EARNINGS PER SHARE (EPS) l ₹349 crore INCOME FROM BUSINESS OPERATIONS l "
            "₹1,613 crore PROFIT AFTER TAX (PAT) l ₹1,23,497 crore NET WORTH")
    got = {(r["metric"], r["value"]) for r in rows(text, "JIOFIN_Annual_Report_FY25.pdf")}
    assert ("Total income", 2079.0) in got
    assert ("Income from business operations", 349.0) in got
    assert ("Profit after tax", 1613.0) in got
    assert ("Earnings per share", 2.54) in got
    assert not any(v == 1594.0 or v == 123497.0 for _, v in got)


def test_eternal_fy26_gaap_highlights():
    text = "FY26 GAAP metrics Revenue** INR 54,364 crore INR 1,208 crore EBITDA INR 366 crore PAT 169% YoY growth"
    got = {(r["metric"], r["value"]) for r in rows(text, "Eternal_Annual_Report_2025-26.pdf")}
    assert ("Total revenue", 54364.0) in got
    assert ("EBITDA", 1208.0) in got
    assert ("Profit after tax", 366.0) in got


def test_reliance_three_year_revenue_table():
    text = "Consolidated financial highlights FY26 FY25 FY24 Revenue 11,75,919 10,71,174 10,00,122"
    got = {(r["period"], r["value"]) for r in rows(text, "RIL_IAR_2026.pdf") if r["metric"] == "Total revenue"}
    assert got == {("FY2026", 1175919.0), ("FY2025", 1071174.0), ("FY2024", 1000122.0)}


def test_bcccl_split_year_header_and_pnl_rows():
    chunks = [
        {"text": "Statement of Profit and Loss Consolidated FY2024-25 FY2023-24", "metadata": {"filename": "BCCL_Annual_Report_2024-25.pdf", "chunk_index": 481}},
        {"text": "Revenue from operations 13,998.45 14,045.34 Total income 14,597.53 14,452.01 Total expenses 12,894.64 12,360.34 Profit before tax 1,702.89 2,091.67 Profit for the year 1,240.19 1,564.46", "metadata": {"filename": "BCCL_Annual_Report_2024-25.pdf", "chunk_index": 482}},
    ]
    got = extract_financial_metrics_from_chunks(chunks)
    revenue = {(r["period"], r["value"]) for r in got if r["metric"] == "Revenue from operations"}
    pat = {(r["period"], r["value"]) for r in got if r["metric"] == "Profit after tax"}
    assert revenue == {("FY2024-25", 13998.45), ("FY2023-24", 14045.34)}
    assert pat == {("FY2024-25", 1240.19), ("FY2023-24", 1564.46)}
    assert any("481-482" in str(r["chunk"]) for r in got if r["metric"] == "Revenue from operations")


def test_adjacent_window_never_crosses_source_documents():
    chunks = [
        {"text": "Consolidated FY2024-25 FY2023-24", "metadata": {"filename": "A.pdf", "chunk_index": 1}},
        {"text": "Revenue 100 90", "metadata": {"filename": "B.pdf", "chunk_index": 2}},
    ]
    got = extract_financial_metrics_from_chunks(chunks)
    assert not any(r["metric"] == "Total revenue" and r["source"] == "A.pdf" for r in got)

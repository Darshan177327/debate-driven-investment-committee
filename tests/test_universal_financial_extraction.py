from tools.financial_dashboard import extract_financial_metrics_from_chunks


def test_multiple_company_layouts():
    chunks = [
        {"text": "Consolidated statement FY 2025-26 FY 2024-25 Revenue from operations 54,364 20,243 Total expenses 53,900 19,700 Profit for the year 366 527", "metadata": {"filename": "Eternal.pdf", "chunk_index": 12}},
        {"text": "FY26 FY25 Revenue from operations 1,075,675 980,136 Profit after tax 81,309 79,020 Total assets 2,100,000 1,900,000", "metadata": {"filename": "Reliance.pdf", "chunk_index": 44}},
        {"text": "Standalone financial statements FY2025 FY2024 Total income 839.28 638.06 Profit after tax 548.91 382.47", "metadata": {"filename": "JioFin.pdf", "chunk_index": 21}},
    ]
    rows = extract_financial_metrics_from_chunks(chunks)
    assert len(rows) == 16
    assert any(r["source"] == "Eternal.pdf" and r["metric"] == "Revenue from operations" and r["period"] == "FY2025-26" and r["value"] == 54364 for r in rows)
    assert any(r["source"] == "Reliance.pdf" and r["metric"] == "Total assets" and r["period"] == "FY2025" and r["value"] == 1900000 for r in rows)
    assert any(r["source"] == "JioFin.pdf" and r["metric"] == "Profit after tax" and r["value"] == 548.91 for r in rows)

from __future__ import annotations

import hashlib
import json
import re
import io
import contextlib
from datetime import datetime
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from agents.committee_controller import run_committee
from tools.multi_company_rag import (
    index_document, list_indexed_companies, get_company_document_count,
    get_all_company_chunks
)
from tools.research_tools import set_active_company
from tools.universal_document_processor import extract_document
from tools.financial_dashboard import extract_financial_metrics_from_chunks

BASE_DIR = Path(__file__).resolve().parent

# Use writable temporary storage on Streamlit Cloud.
# Keep the original project folders on local Windows.
if Path("/mount/src").exists():
    APP_DATA_DIR = (
        Path(tempfile.gettempdir())
        / "debate_driven_investment_committee"
    )
else:
    APP_DATA_DIR = BASE_DIR

UPLOAD_DIR = APP_DATA_DIR / "documents" / "uploads"
LOG_FILE = APP_DATA_DIR / "logs" / "committee_runs.jsonl"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

print("[APP] Upload directory:", UPLOAD_DIR)
print("[APP] Committee log:", LOG_FILE)

SUPPORTED_EXTENSIONS = [
    "pdf", "docx", "xlsx", "xls", "csv", "pptx", "txt", "md",
    "rtf", "html", "json", "xml", "png", "jpg", "jpeg", "tif", "tiff"
]
MAX_UPLOAD_MB = 100

st.set_page_config(
    page_title="Investment Committee | Group 18",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#14283f; --muted:#65768a; --line:#e1e8ef; --paper:#f5f7fa; --teal:#087f78; }
.stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.block-container { max-width:1440px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--ink)!important; font-family:'Manrope',sans-serif!important; letter-spacing:-.035em; }
[data-testid="stSidebar"] { background:#10243a; border-right:1px solid #203a54; }
[data-testid="stSidebar"] * { color:#e9f0f6!important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:8px 10px; border-radius:8px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#1b3852; }
.hero { padding:26px 30px; border-radius:18px; color:#f7fbff; background:linear-gradient(115deg,#10243a 0%,#17465a 62%,#087f78 100%); margin-bottom:22px; }
.hero .eyebrow { color:#9de2d9; text-transform:uppercase; letter-spacing:.16em; font-size:11px; font-weight:700; }
.hero h1 { color:#fff!important; margin:.45rem 0 .35rem 0; font-size:30px; }
.hero p { color:#d7e5ee; margin:0; max-width:760px; }
.kpi { background:#fff; border:1px solid var(--line); border-radius:13px; padding:16px 18px; min-height:100px; }
.kpi-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.1em; font-weight:700; }
.kpi-value { color:var(--ink); font-size:25px; font-weight:800; margin-top:8px; font-family:'Manrope',sans-serif; }
.panel { background:#fff; border:1px solid var(--line); border-radius:14px; padding:20px; margin-bottom:16px; }
.small-muted { color:var(--muted); font-size:13px; }
div.stButton > button, div.stDownloadButton > button, button[kind="primary"] { background:#087f78!important; color:#fff!important; border:1px solid #087f78!important; border-radius:9px!important; font-weight:700!important; min-height:42px; }
div.stButton > button p, div.stDownloadButton > button p { color:#fff!important; }
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"] { background:#fff!important; }
[data-testid="stFileUploader"] { background:#fff; border:1px dashed #aab9c8; border-radius:12px; padding:10px; }
hr { border-color:var(--line); }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower() or "company"

def companies() -> list[str]:
    try:
        return sorted(set(list_indexed_companies()), key=str.casefold)
    except Exception as exc:
        st.error(f"Could not read company index: {exc}")
        return []

def history() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    records = []
    with LOG_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
            except json.JSONDecodeError:
                continue
    return records

def company_documents(company: str) -> list[Path]:
    folder = UPLOAD_DIR / slug(company)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file())

def safe_upload(company: str, uploaded) -> Path:
    ext = Path(uploaded.name).suffix.lower()
    if ext.lstrip(".") not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension: {ext}")
    content = uploaded.getvalue()
    if not content:
        raise ValueError("The file is empty.")
    digest = hashlib.sha256(content).hexdigest()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(uploaded.name).stem).strip("_") or "document"
    folder = UPLOAD_DIR / slug(company)
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{stem}_{digest[:12]}{ext}"
    if not destination.exists():
        destination.write_bytes(content)
    return destination

def get_debate(record: dict) -> str:
    for key in ("debate", "debate_transcript", "transcript"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""

def header(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">GROUP 18 · AI INVESTMENT RESEARCH</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def show_kpi(label: str, value: str, note: str = ""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="small-muted">{note}</div></div>', unsafe_allow_html=True)

def render_dashboard():
    header("Investment intelligence, in one workspace", "A document-grounded research and debate workflow for evaluating companies.")
    comps = companies()
    runs = history()
    latest = runs[-1] if runs else {}
    a,b,c,d = st.columns(4)
    with a: show_kpi("Indexed companies", f"{len(comps)}", "Separate company collections")
    with b: show_kpi("Documents uploaded", f"{sum(len(company_documents(x)) for x in comps)}", "Files stored in this project")
    with c: show_kpi("Committee runs", f"{len(runs)}", "Append-only execution history")
    with d: show_kpi("Latest decision", str(latest.get("final_recommendation","—")), str(latest.get("asset", latest.get("company","No run yet"))))
    st.write("")
    left,right = st.columns([1.35,1])
    with left:
        st.markdown("### Workspace")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if comps:
            frame = pd.DataFrame({"Company":comps, "Indexed chunks":[get_company_document_count(x) for x in comps], "Documents":[len(company_documents(x)) for x in comps]})
            st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.info("No companies indexed. Open **Documents** to add a company report or other business documents.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("### Recent committee activity")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        for item in reversed(runs[-5:]):
            name = item.get("asset", item.get("company","Company"))
            decision = item.get("final_recommendation","—")
            when = item.get("timestamp","")
            st.markdown(f"**{name}** · `{decision}`")
            st.caption(when)
        if not runs:
            st.caption("Your committee executions will appear here.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Investment recommendations are academic outputs, not financial advice.")

def render_companies():
    header("Companies", "Manage indexed companies and inspect how much source material is available to the agents.")
    comps = companies()
    if not comps:
        st.info("No indexed companies yet. Upload a document from the Documents page.")
        return
    selected = st.selectbox("Choose a company", comps, key="company_inspect")
    x,y,z = st.columns(3)
    with x: show_kpi("Indexed chunks", f"{get_company_document_count(selected):,}", "Local vector retrieval")
    with y: show_kpi("Uploaded files", f"{len(company_documents(selected))}", "Files in documents/uploads")
    with z: show_kpi("Collection scope", "Company-only", "No cross-company retrieval")
    st.markdown("### Source documents")
    docs = company_documents(selected)
    if docs:
        st.dataframe(pd.DataFrame([{"File":p.name,"Format":p.suffix.upper().lstrip("."),
                                    "Size":f"{p.stat().st_size/1024:,.1f} KB"} for p in docs]),
                     use_container_width=True, hide_index=True)
    else:
        st.warning("The company has indexed chunks, but no matching upload files were found in the local uploads folder. The collection may have been populated from another path.")

def render_documents():
    header("Document centre", "Upload annual reports, financial statements, presentations and common business documents for company-specific retrieval.")
    left, right = st.columns([1.15, .85])

    with left:
        st.markdown("### Add source material")
        with st.form("upload_form", clear_on_submit=True):
            company = st.text_input("Company / entity name", placeholder="e.g. Reliance Industries Limited")
            files = st.file_uploader("Choose documents", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True)
            submitted = st.form_submit_button("Upload and index documents", type="primary", use_container_width=True)

        if submitted:
            if not company.strip():
                st.error("Enter a company name.")
            elif not files:
                st.error("Choose at least one document.")
            else:
                successes, failures = [], []
                overall = st.progress(0, text="Preparing upload queue")
                overall_caption = st.empty()

                for file_number, uploaded in enumerate(files, start=1):
                    file_label = f"File {file_number}/{len(files)} · {uploaded.name}"
                    with st.status(file_label, expanded=True) as status:
                        st.markdown("#### Document processing")
                        file_info = st.empty()
                        file_info.caption(f"Preparing {uploaded.name} · {uploaded.size / 1024:,.1f} KB")

                        # A single live processing panel: the same panel updates
                        # as the real indexing pipeline reports each stage.
                        stage_panel = st.empty()
                        metric_panel = st.empty()
                        progress_panel = st.empty()
                        activity_panel = st.empty()

                        stage_order = ["extract", "chunk", "embedding", "storage", "verify"]
                        stage_names = {
                            "extract": "Extract document text",
                            "chunk": "Create searchable chunks",
                            "embedding": "Generate semantic embeddings",
                            "storage": "Store vectors in ChromaDB",
                            "verify": "Verify indexed document",
                        }
                        stage_state = {name: "Waiting" for name in stage_order}
                        activity = []
                        stats = {"chunks": 0, "embedded": 0, "stored": 0}
                        active_stage = "extract"

                        def render_live_panel(message=None, phase_fraction=None):
                            rows = []
                            for key in stage_order:
                                state = stage_state[key]
                                if state == "Done":
                                    marker = "🟢"
                                elif state == "Active":
                                    marker = "🟩"
                                else:
                                    marker = "⚪"
                                rows.append(f"{marker} **{stage_names[key]}** · {state}")
                            stage_panel.markdown("\n\n".join(rows))

                            c1, c2, c3 = metric_panel.columns(3)
                            c1.metric("Total chunks", f"{stats['chunks']:,}")
                            c2.metric("Embeddings generated", f"{stats['embedded']:,}")
                            c3.metric("Chunks stored", f"{stats['stored']:,}")

                            if phase_fraction is not None:
                                fraction = max(0.0, min(1.0, float(phase_fraction)))
                                progress_panel.progress(
                                    fraction,
                                    text=f"{stage_names[active_stage]} · {int(fraction * 100)}% of this stage",
                                )
                            elif stage_state["verify"] == "Done":
                                progress_panel.progress(1.0, text="All processing stages completed")
                            else:
                                progress_panel.empty()

                            if message:
                                activity.insert(0, message)
                                activity[:] = activity[:4]
                            if activity:
                                activity_panel.caption("Recent activity  ·  " + "  |  ".join(activity))

                        def show_index_progress(event):
                            nonlocal active_stage
                            stage = event["stage"]
                            message = event.get("message", stage)
                            if stage in ("extract", "extract_done"):
                                active_stage = "extract"
                                stage_state["extract"] = "Active" if stage == "extract" else "Done"
                            elif stage in ("chunk", "chunk_done"):
                                active_stage = "chunk"
                                stage_state["extract"] = "Done"
                                stage_state["chunk"] = "Active" if stage == "chunk" else "Done"
                                if event.get("total") is not None:
                                    stats["chunks"] = int(event["total"])
                            elif stage in ("embedding", "embedding_done"):
                                active_stage = "embedding"
                                stage_state["extract"] = "Done"
                                stage_state["chunk"] = "Done"
                                stage_state["embedding"] = "Active"
                                if event.get("completed") is not None:
                                    stats["embedded"] = int(event["completed"])
                            elif stage in ("storage", "storage_done"):
                                active_stage = "storage"
                                stage_state["extract"] = "Done"
                                stage_state["chunk"] = "Done"
                                stage_state["embedding"] = "Done"
                                stage_state["storage"] = "Active"
                                if event.get("completed") is not None:
                                    stats["stored"] = int(event["completed"])
                            elif stage == "complete":
                                active_stage = "verify"
                                for key in ("extract", "chunk", "embedding", "storage", "verify"):
                                    stage_state[key] = "Done"
                                if event.get("completed") is not None:
                                    stats["stored"] = int(event["completed"])
                            elif stage == "duplicate":
                                for key in ("extract", "chunk", "embedding", "storage"):
                                    stage_state[key] = "Done" if key in ("extract", "chunk") else "Waiting"
                                message = "This exact document was already indexed; no duplicate chunks were added."

                            fraction = None
                            if active_stage == "embedding" and event.get("total"):
                                fraction = stats["embedded"] / max(1, int(event["total"]))
                            elif active_stage == "storage" and event.get("total"):
                                fraction = stats["stored"] / max(1, int(event["total"]))
                            render_live_panel(message, fraction)

                        try:
                            if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                                raise ValueError(f"File exceeds {MAX_UPLOAD_MB} MB.")
                            path = safe_upload(company.strip(), uploaded)
                            result = index_document(
                                company.strip(),
                                str(path),
                                progress_callback=show_index_progress,
                            )
                            if result.get("status") == "Already indexed":
                                stage_panel.info("This document was already indexed. Existing chunks were preserved.")
                            else:
                                stage_state["verify"] = "Done"
                                render_live_panel("Indexing finished successfully.")
                            successes.append((uploaded.name, result))
                            status.update(label=f"{file_label} · {result.get('status', 'Finished')}", state="complete")
                        except Exception as exc:
                            failures.append(f"{uploaded.name}: {exc}")
                            activity_panel.error(f"Processing stopped: {exc}")
                            status.update(label=f"{file_label} · Failed", state="error")

                    overall.progress(file_number / len(files), text=f"Completed {file_number} of {len(files)} files")
                    overall_caption.caption(f"Files completed: {file_number}/{len(files)}")

                for filename, result in successes:
                    st.success(f"{filename}: {result.get('status', 'Indexed')} · {result.get('chunks_indexed', 0):,} new chunks")
                for failure in failures:
                    st.error(failure)
                if successes:
                    st.rerun()

        st.caption("Formats supported by the current processor: PDF, DOCX, XLSX, XLS, CSV, PPTX, TXT, MD, RTF, HTML, JSON, XML and common image formats. OCR on scanned files requires a working Tesseract installation.")

    with right:
        st.markdown("### How indexing works")
        st.markdown('<div class="panel">**1 · Extract** — read text and tables.  \n**2 · Chunk** — split into overlapping evidence passages.  \n**3 · Embed** — generate local FastEmbed vectors in batches.  \n**4 · Store** — write each batch to the company ChromaDB collection.  \n**5 · Retrieve** — agents search the selected company collection.</div>', unsafe_allow_html=True)
        st.markdown("### Existing companies")
        for name in companies():
            st.write(f"**{name}** · {get_company_document_count(name):,} chunks")

def render_financials():
    header(
        "Financial dashboard",
        "Financial charts generated from figures extracted from the company’s indexed annual reports. Each metric is charted separately to avoid mixing unlike values."
    )
    comps = companies()
    if not comps:
        st.info("Upload and index a company document first.")
        return

    company = st.selectbox("Company", comps, key="financial_company")
    total_chunks = get_company_document_count(company)
    st.caption(f"{total_chunks:,} indexed chunks · source: existing company collection")

    with st.spinner(f"Reading indexed evidence for {company}..."):
        try:
            chunks = get_all_company_chunks(company)
            metrics = extract_financial_metrics_from_chunks(chunks)
        except Exception as exc:
            st.error(f"Could not read financial evidence from this company’s index: {exc}")
            return

    if not metrics:
        st.warning(
            "This company is indexed, but no financial figures could be confidently paired with reporting periods. "
            "The source documents remain available to the research agents; no charts have been fabricated."
        )
        with st.expander("Inspect indexed documents"):
            source_rows = {}
            for item in chunks:
                meta = item.get("metadata") or {}
                name = meta.get("filename", "Unknown document")
                source_rows[name] = source_rows.get(name, 0) + 1
            if source_rows:
                st.dataframe(
                    pd.DataFrame([{"Document": name, "Indexed chunks": count} for name, count in sorted(source_rows.items())]),
                    use_container_width=True, hide_index=True
                )
        return

    frame = pd.DataFrame(metrics).copy()
    for col, default in [("scope", "Unspecified"), ("currency", "Unspecified"), ("unit", "Unspecified"), ("source", "Indexed document"), ("chunk", None), ("evidence", "")]:
        if col not in frame.columns:
            frame[col] = default
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["metric", "period", "value"])
    frame = frame.drop_duplicates(subset=["metric", "period", "value", "source", "chunk", "scope", "currency", "unit"])
    if frame.empty:
        st.info("The extractor returned no usable numeric metric-period pairs.")
        return

    def period_key(value):
        nums = re.findall(r"\d+", str(value))
        return (int(nums[0]) if nums else 0, int(nums[1]) if len(nums) > 1 else 0, str(value))

    frame["_period_order"] = frame["period"].map(period_key)
    frame = frame.sort_values("_period_order")

    # Filters deliberately keep unlike reporting bases and units apart.
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        scope_options = sorted(frame["scope"].fillna("Unspecified").astype(str).unique())
        selected_scope = st.selectbox("Reporting basis", scope_options, key="financial_scope")
    with f2:
        currency_options = sorted(frame["currency"].fillna("Unspecified").astype(str).unique())
        selected_currency = st.selectbox("Currency", currency_options, key="financial_currency")
    with f3:
        unit_options = sorted(frame["unit"].fillna("Unspecified").astype(str).unique())
        selected_unit = st.selectbox("Unit", unit_options, key="financial_unit")
    with f4:
        chart_type = st.selectbox("Chart style", ["Bar chart", "Line chart"], key="financial_chart_type")

    filtered = frame[
        (frame["scope"].fillna("Unspecified").astype(str) == selected_scope) &
        (frame["currency"].fillna("Unspecified").astype(str) == selected_currency) &
        (frame["unit"].fillna("Unspecified").astype(str) == selected_unit)
    ].copy()
    if filtered.empty:
        st.info("No figures match these reporting-basis and unit filters.")
        return

    # A metric is always charted against its own reporting periods. Source files
    # remain separate series so duplicate/conflicting reports are visible.
    categories = {
        "Revenue & income": ["revenue", "income", "turnover", "fees", "dividend", "interest"],
        "Profitability": ["profit", "ebitda", "earnings per share", "finance costs", "expenses", "depreciation"],
        "Balance sheet": ["assets", "equity", "liabilities", "borrowings", "cash and cash"],
        "Cash flow": ["cash flow", "operating cash", "investing cash", "financing cash"],
    }
    def category_for(metric):
        low = str(metric).casefold()
        for category, terms in categories.items():
            if any(term in low for term in terms):
                return category
        return "Other metrics"

    filtered["category"] = filtered["metric"].map(category_for)
    periods = sorted(filtered["period"].unique(), key=period_key)
    st.markdown("### Financial performance overview")
    k1, k2, k3 = st.columns(3)
    k1.metric("Extracted figures", f"{len(filtered):,}")
    k2.metric("Financial metrics", f"{filtered['metric'].nunique():,}")
    k3.metric("Reporting periods", f"{len(periods):,}")
    st.caption(f"Showing {selected_scope} · {selected_currency} · {selected_unit}. Figures from different bases or units are intentionally not combined.")

    category_names = [name for name in ["Revenue & income", "Profitability", "Balance sheet", "Cash flow", "Other metrics"] if name in set(filtered["category"])]
    tabs = st.tabs(category_names)
    for tab, category in zip(tabs, category_names):
        with tab:
            category_frame = filtered[filtered["category"] == category]
            metric_names = sorted(category_frame["metric"].unique(), key=str.casefold)
            for offset in range(0, len(metric_names), 2):
                cols = st.columns(2)
                for col, metric_name in zip(cols, metric_names[offset:offset + 2]):
                    with col:
                        metric_frame = category_frame[category_frame["metric"] == metric_name].copy()
                        # Use source as separate series; never add different companies/documents together.
                        chart_frame = metric_frame.pivot_table(
                            index="period", columns="source", values="value", aggfunc="first"
                        )
                        chart_frame = chart_frame.reindex(sorted(chart_frame.index, key=period_key))
                        st.markdown(f"**{metric_name}**")
                        if len(chart_frame.index) >= 2:
                            if chart_type == "Bar chart":
                                st.bar_chart(chart_frame, use_container_width=True)
                            else:
                                st.line_chart(chart_frame, use_container_width=True)
                        elif len(chart_frame.index) == 1:
                            st.metric(str(chart_frame.index[0]), f"{chart_frame.iloc[0, 0]:,.2f}")
                            st.caption("Only one reporting period was confidently extracted; a trend chart needs at least two periods.")
                        else:
                            st.info("No chartable periods for this metric.")
                        with st.expander(f"Evidence · {metric_name}"):
                            evidence_cols = [c for c in ["period", "value", "source", "chunk", "evidence"] if c in metric_frame.columns]
                            st.dataframe(metric_frame[evidence_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "Download extracted figures (CSV)",
        filtered.drop(columns=["_period_order", "category"], errors="ignore").to_csv(index=False),
        file_name=f"{slug(company)}_financial_metrics.csv",
        mime="text/csv",
    )
    with st.expander("Extraction coverage and source documents"):
        coverage = filtered.groupby("source").agg(
            Metrics=("metric", "nunique"),
            Values=("value", "count"),
            Periods=("period", "nunique"),
            Chunks=("chunk", "nunique"),
        ).reset_index().rename(columns={"source": "Document"})
        st.dataframe(coverage, use_container_width=True, hide_index=True)
        st.caption("Charts are generated from extracted report figures, not chart images copied from PDFs. Open each Evidence panel to verify the original text and chunk reference.")

def render_committee():
    header("Committee room", "Run the Research → Bull → Bear → Rebuttal → Judge workflow, followed by the Risk Agent veto check.")
    comps = companies()
    if not comps:
        st.info("Index at least one company document before running the committee.")
        return
    company = st.selectbox("Company to evaluate", comps, key="committee_company")
    st.caption(f"{get_company_document_count(company):,} indexed chunks · company-specific retrieval")
    if st.button("Run investment committee", type="primary", use_container_width=True):
        trace = io.StringIO()
        with st.spinner(f"Agents are evaluating {company}..."):
            try:
                set_active_company(company)
                with contextlib.redirect_stdout(trace):
                    result = run_committee(company)
                st.session_state["committee_result"] = result
                st.session_state["committee_trace"] = trace.getvalue()
                st.session_state["committee_debate"] = result.get("debate","")
                st.session_state["last_committee_company"] = company
            except Exception as exc:
                st.error(f"Committee execution failed: {exc}")
                st.code(trace.getvalue())
    result = st.session_state.get("committee_result")
    if result and st.session_state.get("last_committee_company") == company:
        risk = result.get("risk_result") or {}
        st.markdown("### Committee decision")
        st.markdown(f"## {result.get('final_recommendation','HOLD')}")
        st.write(result.get("decision_reason",""))
        a,b,c = st.columns(3)
        with a: show_kpi("Judge", result.get("judge_recommendation","—"), "Initial decision")
        with b: show_kpi("Risk score", f"{risk.get('risk_score','—')}/100", "Heuristic screening score")
        with c: show_kpi("Risk veto", str(risk.get("veto","Unavailable")), f"Threshold {risk.get('threshold','—')}")
        tabs = st.tabs(["Judge analysis","Debate transcript","Risk assessment","Execution trace"])
        with tabs[0]: st.write(result.get("judge_analysis","No judge output"))
        with tabs[1]: st.text_area("Debate", st.session_state.get("committee_debate",""), height=360)
        with tabs[2]:
            st.write(risk.get("explanation","No risk explanation available"))
            st.caption("The risk score is a fixed-weight keyword heuristic, not a calibrated probability or independent audit.")
        with tabs[3]: st.text_area("Trace", st.session_state.get("committee_trace",""), height=300)

def render_history():
    header("Run history", "Review previous committee decisions and inspect the evidence debate saved with each execution.")
    records = history()
    if not records:
        st.info("No committee runs have been logged yet.")
        return
    frame = pd.DataFrame([{
        "Timestamp":r.get("timestamp",""),
        "Company":r.get("asset",r.get("company","Unknown")),
        "Judge":(r.get("judge_result") or {}).get("recommendation",""),
        "Final":r.get("final_recommendation",""),
        "Risk score":(r.get("risk_result") or {}).get("risk_score","")
    } for r in records])
    st.dataframe(frame.iloc[::-1], use_container_width=True, hide_index=True)
    options = list(range(len(records)-1,-1,-1))
    chosen = st.selectbox("Open run", options, format_func=lambda i: f"{records[i].get('asset',records[i].get('company','Company'))} · {records[i].get('final_recommendation','—')} · {records[i].get('timestamp','')}")
    record = records[chosen]
    st.markdown("### Saved decision")
    st.write(record.get("decision_reason",""))
    with st.expander("Debate transcript", expanded=True):
        st.text(record.get("debate","No transcript saved."))
    with st.expander("Judge and risk outputs"):
        st.json({"judge_result":record.get("judge_result"),"risk_result":record.get("risk_result")})

def render_about():
    header("System & methodology", "Understand the architecture, evidence boundaries and limitations before interpreting an AI-generated recommendation.")
    st.markdown("### Agent workflow")
    st.markdown('<div class="panel">**Research Agent** retrieves source evidence → **Bull Agent** builds the positive thesis → **Bear Agent** challenges it → **Rebuttal Agent** responds to opposing arguments → **Judge Agent** weighs the debate → **Risk Agent** applies a predefined veto rule → **Committee Controller** records the outcome.</div>', unsafe_allow_html=True)
    st.markdown("### Retrieval and storage")
    st.write("Documents are extracted by the universal document processor, chunked, embedded locally with FastEmbed (BAAI/bge-small-en-v1.5), and stored in company-specific ChromaDB collections. Gemini is used for agent reasoning, not for document embeddings.")
    st.markdown("### Important limitations")
    st.warning("The risk score is a keyword-based academic heuristic. Automated financial extraction is a first-pass parser. Agents can make mistakes; verify important claims against the original reports. Outputs are not investment advice.")
    st.markdown("### Project technologies")
    st.write("Python 3.11 · Streamlit · Microsoft Agent Framework · Gemini · FastEmbed · ChromaDB · PyMuPDF · Pandas")

if "committee_result" not in st.session_state:
    st.session_state["committee_result"] = None

with st.sidebar:
    st.markdown("## ◈ INVESTMENT\n### COMMITTEE")
    st.caption("GROUP 18 · RESEARCH WORKSPACE")
    page = st.radio("NAVIGATION", ["Overview","Companies","Documents","Financials","Committee room","Run history","System & methodology"], label_visibility="collapsed")
    st.divider()
    st.caption("LOCAL DOCUMENT INDEX")
    for company_name in companies():
        st.caption(f"▸ {company_name} · {get_company_document_count(company_name):,} chunks")
    st.divider()
    st.caption("Gemini reasoning · Local embeddings")

routes = {
    "Overview":render_dashboard,
    "Companies":render_companies,
    "Documents":render_documents,
    "Financials":render_financials,
    "Committee room":render_committee,
    "Run history":render_history,
    "System & methodology":render_about,
}
routes[page]()

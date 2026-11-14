from __future__ import annotations

import hashlib
import json
import re
import io
import contextlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from agents.committee_controller import run_committee
from tools.multi_company_rag import (
    index_document, list_indexed_companies, get_company_document_count
)
from tools.research_tools import set_active_company
from tools.universal_document_processor import extract_document
from tools.financial_dashboard import extract_financial_metrics

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "documents" / "uploads"
LOG_FILE = BASE_DIR / "logs" / "committee_runs.jsonl"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

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
    left,right = st.columns([1.15,.85])
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
                progress = st.progress(0)
                successes, failures = [], []
                for i, uploaded in enumerate(files):
                    try:
                        if uploaded.size > MAX_UPLOAD_MB*1024*1024:
                            raise ValueError(f"File exceeds {MAX_UPLOAD_MB} MB.")
                        path = safe_upload(company.strip(), uploaded)
                        result = index_document(company.strip(), str(path))
                        successes.append((uploaded.name, result))
                    except Exception as exc:
                        failures.append(f"{uploaded.name}: {exc}")
                    progress.progress((i+1)/len(files))
                for filename,result in successes:
                    st.success(f"{filename}: {result.get('status','Indexed')} · {result.get('chunks_indexed',0)} new chunks")
                for failure in failures:
                    st.error(failure)
                if successes:
                    st.rerun()
        st.caption("Formats supported by the current processor: PDF, DOCX, XLSX, XLS, CSV, PPTX, TXT, MD, RTF, HTML, JSON, XML and common image formats. OCR on scanned files requires a working Tesseract installation.")
    with right:
        st.markdown("### How indexing works")
        st.markdown('<div class="panel">**1 · Extract** — text and tables are read from the uploaded file.  \n**2 · Split** — content is divided into overlapping evidence chunks.  \n**3 · Embed locally** — FastEmbed creates 384-dimensional vectors.  \n**4 · Store** — ChromaDB keeps a separate collection for each company.  \n**5 · Retrieve** — agents search only the selected company collection.</div>', unsafe_allow_html=True)
        st.markdown("### Existing companies")
        for name in companies():
            st.write(f"**{name}** · {get_company_document_count(name):,} chunks")

def render_financials():
    header("Financial dashboard", "Charts are generated only from financial figures that can be extracted from the company's uploaded source documents.")
    comps = companies()
    if not comps:
        st.info("Upload and index a financial document first.")
        return
    company = st.selectbox("Company", comps, key="financial_company")
    docs = company_documents(company)
    if not docs:
        st.warning("No original files are available in the uploads folder for this company. The vector index remains available for agent retrieval, but this chart parser needs the source file.")
        return
    selected_docs = st.multiselect("Source documents", [p.name for p in docs], default=[p.name for p in docs])
    metrics = []
    errors = []
    for path in docs:
        if path.name not in selected_docs:
            continue
        try:
            extracted = extract_document(str(path))
            metrics.extend(extract_financial_metrics(extracted["full_text"], path.name))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if errors:
        for error in errors: st.warning(error)
    if not metrics:
        st.info("No comparable two-period financial statement rows were confidently recognised in the selected files. No chart has been fabricated. Try an annual report PDF or a structured XLSX/CSV statement.")
        return
    frame = pd.DataFrame(metrics)
    frame = frame.drop_duplicates(subset=["metric","period","value","source"])
    st.caption("Extracted values are in the units printed in the source. Check each source before relying on a chart; this is an automated first-pass parser.")
    metric_names = sorted(frame["metric"].unique())
    chosen = st.selectbox("Financial metric", metric_names)
    subset = frame[frame["metric"] == chosen]
    pivot = subset.pivot_table(index="period", columns="source", values="value", aggfunc="first")
    st.markdown(f"### {chosen}")
    st.bar_chart(pivot, use_container_width=True)
    st.dataframe(subset.sort_values(["source","period"]), use_container_width=True, hide_index=True)
    st.download_button("Download extracted figures (CSV)", subset.to_csv(index=False), file_name=f"{slug(company)}_extracted_financials.csv", mime="text/csv")
    st.warning("Automated extraction can misread table layouts, units, or standalone versus consolidated statements. Verify the displayed values against the cited source document.")

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

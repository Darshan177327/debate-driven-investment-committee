# Debate-Driven Investment Committee

## 1. Overview

A multi-agent investment-analysis system that examines company documents, builds opposing investment cases, conducts rebuttals, and produces a **BUY, HOLD, or AVOID** recommendation. A Risk Agent applies a conservative veto rule, and every completed run is recorded in an append-only JSONL log.

The demonstration run uses **Jio Financial Services Limited**. The Streamlit interface supports selecting companies and uploading company documents for indexing and analysis.

## 2. Architecture and Agent Responsibilities

The system uses **Microsoft Agent Framework (MAF)** with Google Gemini for agent reasoning. The Committee Controller coordinates the workflow:

| Agent | Responsibility | Python tools used |
|---|---|---|
| Research | Retrieves relevant company evidence and prepares the research context. | Semantic document search; keyword document search |
| Bull | Builds the positive investment case from retrieved evidence. | Semantic document search; keyword document search |
| Bear | Builds the opposing case and identifies risks. | Semantic document search; keyword document search |
| Rebuttal | Produces a Bull-side rebuttal to the Bear case and a Bear-side rebuttal to the Bull case. | Semantic document search; keyword document search |
| Judge | Weighs the debate against retrieved evidence and recommends BUY, HOLD, or AVOID. | Semantic document search; keyword document search |
| Risk | Identifies configured risk indicators and calculates a weighted score. | Risk-indicator identification; risk-score calculation |

The controller passes the debate to the Judge and Risk Agent, applies the veto rule, and sends the result to the run logger. The architecture source is `docs/architecture.mmd`.

## 3. Retrieval, Memory, and Document Processing

- **Document ingestion:** `tools/universal_document_processor.py` extracts content from PDF, DOCX, XLSX/XLS, CSV, PPTX, TXT/MD, RTF, HTML, JSON, and XML files. Image OCR depends on a separately installed Tesseract executable.
- **Embeddings:** FastEmbed using `BAAI/bge-small-en-v1.5` creates local 384-dimensional embeddings. This avoids using Gemini embedding requests.
- **Vector store:** ChromaDB persists company-specific collections under `memory/vector_store`.
- **Retrieval:** Agents use semantic search and keyword search through `tools/research_tools.py`.
- **Company separation:** `tools/multi_company_rag.py` keeps indexed documents in company-specific collections to reduce cross-company evidence mixing.

Retrieved evidence is identified by source filename and chunk number. A chunk number is **not** a PDF page number. The risk score is a predefined weighted heuristic for this academic prototype, not a statistically calibrated probability or professional risk rating.

## 4. Risk Rule, Logging, and Failure Handling

The Risk Agent checks configured indicators including fair-value gains, declining interest income, subsidiary impairment indicators, missing asset-quality information, and rising expenses. The configured veto threshold is **70/100**.

- If a valid risk veto is triggered, a Judge **BUY** is changed to **HOLD**.
- A Judge **AVOID** remains **AVOID**.
- If the risk result or veto status is unavailable or invalid, the controller defaults to **HOLD**.
- If the debate or Judge step fails, the controller uses a conservative **HOLD** fallback.
- A logging failure is reported and does not change the investment decision.

Runs are appended to `logs/committee_runs.jsonl`, including the company, debate transcript, Judge result, risk result, final recommendation, and decision reason. Existing records are not intentionally overwritten.

## 5. Technology

Python 3.11 · Microsoft Agent Framework · Google Gemini (`gemini-3.5-flash-lite`) · Streamlit · FastEmbed · ChromaDB · PyMuPDF · python-dotenv · pytest · Jupyter Notebook.

## 6. Setup and Run

Run these commands in PowerShell from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install agent-framework agent-framework-gemini google-genai streamlit chromadb fastembed pymupdf python-dotenv pydantic jupyter pytest python-docx openpyxl xlrd python-pptx striprtf beautifulsoup4 pillow pandas
```

Create a `.env` file in the project root and add your own Gemini API key:

```text
GOOGLE_API_KEY=your_api_key_here
```

Keep `.env` private. Do not commit or share API keys.

Start the web application:

```powershell
streamlit run app.py
```

Open the local address printed by Streamlit in your browser. Upload and index a company's supported document before running its committee analysis.

## 7. Demo and Tests

- **Runnable demonstration:** `Group18_Debate_Driven_Investment_Committee_Demo_FIXED.ipynb`. Run the notebook cells in order. The notebook invokes the controller in a subprocess to avoid nested event-loop conflicts in Jupyter.
- **Decision-logic tests:** run `python -m pytest tests/test_committee.py -v`. The four current tests cover BUY without veto, BUY overridden by veto, AVOID without veto, and missing risk assessment defaulting to HOLD.
- **Framework/tool check:** `python tests/test_agent_framework.py` (if using this standalone check).
- **Execution evidence:** a successful end-to-end notebook run was recorded for Jio Financial Services. It showed both rebuttal outputs, a Judge recommendation of HOLD, a heuristic risk score of 60/100 (below the threshold of 70), and a final HOLD decision. The run was appended to the JSONL log.

The four automated tests validate controller decision logic; they are not three full end-to-end scenario tests. The successful notebook run is the current end-to-end demonstration.

## 8. Academic Use

This project demonstrates evidence retrieval, multi-agent debate, risk-based decision control, and execution logging. Its output is for academic demonstration only and is **not investment advice**.


# Debate-Driven Investment Committee

🚀 **Live Website:** [Open the Investment Committee Application](https://debate-driven-investment-committee.streamlit.app/)

💻 **Source Code:** [GitHub Repository](https://github.com/Darshan177327/debate-driven-investment-committee)

An AI-powered multi-agent investment analysis system using Microsoft Agent Framework, Google Gemini, RAG and Streamlit.

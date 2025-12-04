# Intelligent Underwriting Workbench 🛡️

**An Agentic AI Co‑Pilot for Commercial Risk Assessment**

---

## 📌 Project Overview

The **Intelligent Underwriting Workbench** is an AI-powered application built to solve the "unstructured data overload" problem in commercial insurance underwriting. Rather than spending hours manually reading PDF applications, loss runs and credit reports, this workbench ingests and synthesizes risk data using a Multi‑Agent architecture and produces:

* A consistent **Risk Score (0–10)**
* A clear **Recommendation** (Approve / Decline)
* An **Auto-drafted decision letter** with rationale

It is designed for privacy-first local deployment (Ollama) and enforces structured outputs using Pydantic.

---

## 🚀 Key Features

* **Robust PDF Ingestion** — Custom regex pipeline to repair column/table text extraction.
* **Multi‑Agent Analysis** — Parallel Claims, Profile, and External agents for speed and accuracy.
* **Structured Guardrails** — Pydantic schemas force LLM responses into reliable JSON.
* **Privacy‑First** — Runs locally on Ollama; applicant data never leaves the host.
* **Advanced Analytics** — Interactive radar charts and explainability dashboards (Plotly).
* **Automated Decisioning** — Auto-draft formal decision letters tailored to flagged risk factors.

---

## 🏗️ System Architecture

High-level layers:

1. **Ingress**: Streamlit-based UI — upload PDF or paste text.
2. **Application**: Cleaning, text extraction & pre-processing.
3. **Intelligence Engine** (The Brain):

   * **Orchestrator** — dispatches tasks to agents (concurrent execution).
   * **Specialist Agents** — Claims, Profile, External (each evaluates a focused sub-domain).
   * **Synthesis Manager** — merges agent outputs, applies business rules & guidelines (RAG).
4. **Data & Validation**: Pydantic models validate outputs before presenting in UI.
5. **Presentation**: Streamlit dashboard with gauges, radar charts, and a Deep Dive view.

---

## 🧩 Technology Stack

* **Frontend**: Streamlit (Python)
* **Orchestration**: LangChain (RunnableParallel for parallel agents)
* **LLM**: Ollama (Llama3 local inference)
* **Validation**: Pydantic
* **Visualization**: Plotly (gauges, radar charts)
* **PDF Parsing**: pypdf + custom regex cleanup

---

## ⚙️ Installation & Setup

> Tested on Python 3.10+ and Ollama running locally.

1. Clone the repository

```bash
git clone https://github.com/yourusername/underwriting-workbench.git
cd underwriting-workbench
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

> Ensure `requirements.txt` contains at least:
>
> ```text
> streamlit
> langchain
> langchain-ollama
> pydantic
> plotly
> pypdf
> streamlit-option-menu
> ```

3. Pull the model to Ollama (local)

```bash
ollama pull llama3
```

4. Run the app

```bash
streamlit run app.py
```

---

## 📸 Usage

1. **Upload**: Drag & drop a commercial insurance application (PDF) or paste raw text into the Streamlit UI.
2. **Analyze**: Agents run in parallel and extract Claims, Credit, and Legal signals.
3. **Review**: Dashboard displays the aggregated **Risk Score** and a quick decision card.
4. **Deep Dive**: Inspect agent-level reasoning, sources, and the Pydantic-validated JSON.
5. **Act**: Copy the auto-generated decision email/letter for policy communication.

---

## 🎛️ Notable Implementation Details

* **Multi-Agent Orchestration**: Use LangChain `RunnableParallel` to run three specialist flows concurrently, then merge results in a Synthesis Manager.
* **Pydantic Contracts**: Define tight schemas for `AgentOutput`, `RiskFactors[]`, and the final `UnderwritingDecision` to avoid hallucinations.
* **PDF Repair Pipeline**: Heuristic regex transforms correct broken column merges from PDF text extraction.
* **Explainability**: Each agent stores a short human-readable rationale and the citations (text spans) it used.
* **Local-Only LLM**: Ollama + Llama3 for local inference and data privacy.

---

## 🔮 Roadmap

* **Phase 2**: Connect a Vector DB (e.g., FAISS/Chroma) for full RAG with historical decisions.
* **Phase 3**: Human‑in‑the‑Loop feedback and supervised fine‑tuning of agent prompts and scoring.
* **Phase 4**: API integration with policy admin & core insurance systems for automated binding.

---

## 👥 Team

* Gurupreet Dhande
* Khushi Dekate
* Arnav Kalambe

---

## 📄 Suggested Repository Structure

```
underwriting-workbench/
├─ app.py                   # Streamlit entrypoint
├─ README.md
├─ requirements.txt
├─ pipelines/
│  ├─ pdf_cleanup.py
│  ├─ ingestion.py
│  └─ validators.py         # pydantic models
├─ agents/
│  ├─ claims_agent.py
│  ├─ profile_agent.py
│  └─ external_agent.py
├─ orchestrator/
│  └─ runnables.py         # langchain runnables + parallelism
├─ visualizations/
│  └─ charts.py             # plotly radar/gauge helpers
└─ tests/
   └─ test_pipelines.py
```

---

## ✅ Contributing

Contributions are welcome. Please open issues for feature requests or bug reports. For code changes, send a PR with tests and update `requirements.txt` if you add dependencies.

---

## 📜 License

Specify a license (e.g., MIT) and include a `LICENSE` file in the repo.

---

# frontend/streamlit_app.py
# Merged and enhanced Underwriting Workbench
import streamlit as st
from streamlit_option_menu import option_menu
import json, io, os, time, math, tempfile, re
from pydantic import BaseModel, Field
from typing import Literal
from io import BytesIO
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- Optional heavy deps (safely imported) ---
SHAP_AVAILABLE = False
ML_MODEL_PATH = os.getenv("UM_XGB_MODEL", "backend/artifacts/xgb_model.joblib")
ml_model = None
try:
    import joblib
    import shap
    SHAP_AVAILABLE = True
    if os.path.exists(ML_MODEL_PATH):
        try:
            ml_model = joblib.load(ML_MODEL_PATH)
        except Exception:
            ml_model = None
except Exception:
    SHAP_AVAILABLE = False
    ml_model = None

# --- LangChain / Ollama try import, with graceful fallback ---
HAS_OLLAMA = False
try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage
    from langchain_core.runnables import RunnableParallel
    from pypdf import PdfReader
    HAS_OLLAMA = True
except Exception:
    # Ollama or langchain not available — app will fall back to rule-based processing
    HAS_OLLAMA = False
    # ensure PdfReader available as fallback
    try:
        from pypdf import PdfReader
    except Exception:
        PdfReader = None

# ------------------ Pydantic Schemas ------------------
class SpecialistRiskOutput(BaseModel):
    risk_score: int = Field(..., ge=1, le=10, description="1-10")
    summary: str = Field(..., description="single-sentence justification")

class FinalUnderwritingResult(BaseModel):
    final_risk_score: float = Field(..., ge=0.0, le=10.0, description="0.0-10.0")
    recommendation: Literal["Approve", "Refer to Senior Underwriter", "Decline"] = Field(...)
    justification: str = Field(..., description="2-3 sentence justification")

# ------------------ Agent Factory (from original code2) ------------------
def create_specialist_chain(name: str, system_prompt: str, llm):
    format_instructions = (
        "You MUST provide your response as a valid JSON object. "
        "Keys: 'risk_score' (1-10 int) and 'summary' (string). No extra text."
    )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        SystemMessage(content=format_instructions),
        ("human", "Analyze this applicant data:\n\n{input_data}"),
    ])
    return prompt | llm.with_structured_output(SpecialistRiskOutput, method="json_mode")

def create_letter_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional Insurance Underwriting Assistant. Write a formal decision email/letter to the broker."),
        ("human", "Applicant Data: {input_data}\n\nDecision: {decision}\n\nReasoning: {reasoning}\n\nWrite a polite, professional email regarding this decision.")
    ])
    return prompt | llm

@st.cache_resource
def build_underwriting_agent():
    """Build agents if Ollama/langchain available; otherwise return None."""
    if not HAS_OLLAMA:
        return None, None, None
    llm = ChatOllama(model="llama3:8b", temperature=0.0)
    # specialists
    claims_chain = create_specialist_chain("Claims", "You are the Claims Risk Specialist. Analyze claims frequency, severity, and recency.", llm)
    profile_chain = create_specialist_chain("Profile", "You are the Profile Specialist. Analyze credit score, business age, and industry risk.", llm)
    external_chain = create_specialist_chain("External", "You are the External Reports Specialist. Analyze safety violations, legal issues, and public records.", llm)

    # synthesis
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Lead Underwriter. Synthesize the reports to make a final decision."),
        ("system", "Adhere STRICTLY to these Underwriting Guidelines:\n{guidelines}"),
        ("system", "Output JSON with keys: 'final_risk_score' (0.0-10.0), 'recommendation'(Approve/Refer/Decline), 'justification'."),
        ("human", "--- Reports ---\nCLAIMS: {claims}\nPROFILE: {profile}\nEXTERNAL: {external}\n\nDecision:")
    ])
    synthesis_chain = synthesis_prompt | llm.with_structured_output(FinalUnderwritingResult, method="json_mode")

    # parallel processing chain
    processing_chain = RunnableParallel(
        claims=claims_chain,
        profile=profile_chain,
        external=external_chain,
        guidelines=lambda x: x.get('guidelines', 'Standard underwriting principles apply.'),
        input_data=lambda x: x['input_data']
    )

    letter_gen = create_letter_chain(llm)
    return processing_chain, synthesis_chain, letter_gen

# Attempt to build agent (cached)
proc_chain, synth_chain, letter_gen = build_underwriting_agent()

# ------------------ Fallback rule-based scorer (from code1) ------------------
def rule_score(app):
    score = 0.0
    if app.get('age', 99) < 25:
        score += 15
    elif app.get('age', 99) > 60:
        score += 10
    cs = app.get('credit_score', 700)
    if cs < 580:
        score += 25
    elif cs < 650:
        score += 12
    elif cs < 700:
        score += 6
    num = len(app.get('prior_claims', []))
    total = sum(c.get('amount',0) for c in app.get('prior_claims', []))
    score += num * 8
    if total > 10000:
        score += 12
    elif total > 3000:
        score += 6
    score += app.get('driving_violations_last_5y', 0) * 6
    if 'driver' in app.get('occupation','').lower():
        score += 8
    if app.get('vehicle_age',0) > 5:
        score += 4
    if app.get('location_risk') == 'High':
        score += 10
    elif app.get('location_risk') == 'Moderate':
        score += 5
    score += app.get('external_reports', {}).get('public_records_flags', 0) * 6
    score += app.get('external_reports', {}).get('social_media_flags', 0) * 4
    return max(0, min(100, round((score / 120) * 100, 2))), {
        'raw_points': score,
        'num_claims': num,
        'total_claims_amount': total
    }

# ML wrapper if model available
def ml_score(app):
    if ml_model is None:
        return 0.5 * 100, {'prob': 0.5}
    feat = {
        'age': app.get('age', 40),
        'credit_score': app.get('credit_score', 700),
        'num_claims': len(app.get('prior_claims', [])),
        'total_claims_amt': sum(c.get('amount',0) for c in app.get('prior_claims', [])),
        'violations': app.get('driving_violations_last_5y', 0),
        'vehicle_age': app.get('vehicle_age', 0),
        'location_high': 1 if app.get('location_risk') == 'High' else 0,
        'location_mod': 1 if app.get('location_risk') == 'Moderate' else 0
    }
    X = pd.DataFrame([feat])
    prob = float(ml_model.predict_proba(X)[:,1][0])
    return prob*100, {'prob': prob}

def shap_explain(app, topk=5):
    if ml_model is None or not SHAP_AVAILABLE:
        return None
    feat = {
        'age': app.get('age', 40),
        'credit_score': app.get('credit_score', 700),
        'num_claims': len(app.get('prior_claims', [])),
        'total_claims_amt': sum(c.get('amount',0) for c in app.get('prior_claims', [])),
        'violations': app.get('driving_violations_last_5y', 0),
        'vehicle_age': app.get('vehicle_age', 0),
        'location_high': 1 if app.get('location_risk') == 'High' else 0,
        'location_mod': 1 if app.get('location_risk') == 'Moderate' else 0
    }
    X = pd.DataFrame([feat])
    explainer = shap.TreeExplainer(ml_model)
    vals = explainer.shap_values(X)[0]
    features = X.columns.tolist()
    pairs = sorted(zip(features, vals.tolist()), key=lambda x: -abs(x[1]))
    return [{'feature': f, 'impact': float(v)} for f, v in pairs[:topk]]

# ------------------ PDF extraction & TF-IDF RAG ------------------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text.append(t)
        raw = "\n".join(text)
        if not raw:
            return ""
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        raw = raw.replace("●", "\n• ").replace("•", "\n• ")
        raw = re.sub(r'[ \t ]+', ' ', raw)
        raw = re.sub(r'\s*\n\s*', '\n', raw)
        raw = re.sub(r'([^\n])\n([^\n])', r'\1 \2', raw)
        raw = re.sub(r'([^\n])\n([^\n])', r'\1 \2', raw)
        raw = re.sub(r'\n{2,}', '\n\n', raw)
        return raw.strip()
    except Exception:
        return ""

from sklearn.feature_extraction.text import TfidfVectorizer
def build_paragraph_index(text:str):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return [], None
    vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2)).fit(paras)
    mat = vect.transform(paras)
    return paras, (vect, mat)

def query_paragraphs(index_tuple, query, top_k=3):
    if not index_tuple:
        return []
    vect, mat = index_tuple
    qv = vect.transform([query])
    scores = (mat @ qv.T).toarray().ravel()
    idxs = np.argsort(-scores)[:top_k]
    return [(int(i), scores[int(i)]) for i in idxs if scores[int(i)]>0]

# ------------------ Visual helpers ------------------
def risk_gauge(score: float, title="Risk Score (0-10)"):
    s10 = score/10.0
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = s10,
        number={'valueformat':".1f"},
        domain = {'x':[0,1], 'y':[0,1]},
        title={'text': title},
        gauge={
            'axis': {'range':[0,10]},
            'steps':[{'range':[0,4], 'color':'#22C55E'},{'range':[4,7], 'color':'#F59E0B'},{'range':[7,10], 'color':'#EF4444'}],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': s10}
        }
    ))
    return fig

def get_radar_chart(specialists_data):
    categories = ['Claims History', 'Business Profile', 'External Factors']
    try:
        scores = [
            specialists_data['claims'].risk_score,
            specialists_data['profile'].risk_score,
            specialists_data['external'].risk_score
        ]
    except Exception:
        scores = [5,5,5]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Risk Profile',
        line_color='#3B82F6',
        fillcolor='rgba(59, 130, 246, 0.4)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color='gray')),
            angularaxis=dict(tickfont=dict(size=14, color='white'))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        showlegend=False
    )
    return fig

# ------------------ Persistence: local audit log ------------------
AUDIT_PATH = os.path.join("frontend_artifacts")
os.makedirs(AUDIT_PATH, exist_ok=True)
AUDIT_FILE = os.path.join(AUDIT_PATH, "audit_log.jsonl")

def append_audit(applicant_id, payload):
    entry = {'timestamp': datetime.utcnow().isoformat()+"Z", 'applicant_id': applicant_id, 'decision': payload}
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_audit(applicant_id=None):
    if not os.path.exists(AUDIT_FILE):
        return []
    out = []
    with open(AUDIT_FILE,"r") as f:
        for line in f:
            try:
                j = json.loads(line)
                if applicant_id is None or j.get('applicant_id') == applicant_id:
                    out.append(j)
            except:
                continue
    return out

def draw_timeline(audit_records):
    if not audit_records:
        st.info("No audit records")
        return
    df = pd.DataFrame(audit_records)
    df['ts'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('ts')
    fig = go.Figure()
    # safe handling: decision may contain risk_score or nested payload
    def score_for_plot(d):
        dec = d.get('decision', {})
        if isinstance(dec, dict):
            return dec.get('risk_score') or dec.get('final_risk_score') or 0
        return 0
    fig.add_trace(go.Scatter(x=df['ts'], y=df.apply(score_for_plot, axis=1),
                             mode='markers+lines', text=df['decision'].apply(lambda d: json.dumps(d)[:200])))
    fig.update_layout(title="Audit Timeline (score over time)", xaxis_title="Time", yaxis_title="Risk Score (0-100)")
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Streamlit UI ------------------
st.set_page_config(layout="wide", page_title="Underwriting Workbench+", page_icon="🛡️")
st.markdown("<style>body { background-color: #081028; color: #E6EEF8; }</style>", unsafe_allow_html=True)

with st.sidebar:
    selected_page = option_menu(
        menu_title="Workbench+",
        options=["Dashboard", "Deep Dive", "Live Co-Pilot", "Batch Scoring", "Counterfactuals", "Settings"],
        icons=["speedometer2","clipboard-data-fill","chat-dots-fill","cloud-arrow-up","vector-pen","sliders"],
        menu_icon="shield-shaded",
        default_index=0,
        styles={
            "container": {"background-color": "#071029"},
            "icon": {"color": "#60A5FA", "font-size": "18px"},
            "nav-link": {"font-size": "15px", "--hover-color": "#0b1a2b"},
            "nav-link-selected": {"background-color": "#0b2a4a"},
        }
    )
    st.markdown("---")
    st.markdown("**Team:** Gurupreet Dhande, Khushi Dekate, Arnav Kalambe")
    st.markdown("**Advanced:** Counterfactuals • Batch Scoring • TF-IDF RAG • Ensemble Tuner")
    st.markdown("---")

# ---------- DASHBOARD ----------
if selected_page == "Dashboard":
    st.title("🛡️ Intelligent Underwriting Workbench")
    col_input, col_output = st.columns([1, 1.2], gap="large")
    with col_input:
        st.subheader("1. Ingestion")
        input_method = st.radio("Input Source", ["Text Paste", "PDF Upload"], horizontal=True)
        raw_text = ""
        if input_method == "Text Paste":
            raw_text = st.text_area("Applicant Data", height=300, placeholder="Paste unstructured applicant notes here...")
        else:
            uploaded_file = st.file_uploader("Upload Application (PDF)", type="pdf")
            if uploaded_file:
                raw_text = extract_text_from_pdf_bytes(uploaded_file.read())
                st.success("PDF Extracted Successfully")
                with st.expander("View Raw Text"):
                    st.caption((raw_text or "")[:1000] + ("..." if raw_text and len(raw_text)>1000 else ""))
        if st.button("🚀 Run Risk Analysis", type="primary", use_container_width=True):
            if not raw_text:
                st.warning("Please provide input data.")
            else:
                guidelines = st.sidebar.text_area("Underwriting Guidelines", value="Decline businesses with >2 claims in 3 years.\nRefer construction companies with credit score < 650.", height=120)
                with st.spinner("🤖 Agents working in parallel..."):
                    try:
                        if proc_chain and synth_chain:
                            specialists_out = proc_chain.invoke({"input_data": raw_text, "guidelines": guidelines})
                            final_out = synth_chain.invoke({
                                "claims": specialists_out['claims'],
                                "profile": specialists_out['profile'],
                                "external": specialists_out['external'],
                                "guidelines": guidelines
                            })
                            st.session_state.current_result = final_out
                            st.session_state.current_specialists = specialists_out
                            append_audit(f"LLM_{int(time.time())}", {'final_risk_score': final_out.final_risk_score*10, 'recommendation': final_out.recommendation, 'provenance':'llm'})
                        else:
                            # fallback rule-based parse & scoring
                            name_search = re.search(r"Name[:\\s]+([A-Za-z ]{2,50})", raw_text) or re.search(r"Applicant[:\\s]+([A-Za-z ]{2,50})", raw_text)
                            name = name_search.group(1).strip() if name_search else "Unknown Applicant"
                            applicant = {
                                'applicant_id': f"PDF_{int(time.time())}",
                                'name': name,
                                'age': 40,
                                'occupation': 'Unknown',
                                'annual_income': 30000,
                                'credit_score': 640,
                                'prior_claims': [],
                                'policy_type': 'Auto',
                                'vehicle_age': 5,
                                'driving_violations_last_5y': 0,
                                'location_risk': 'Moderate',
                                'external_reports': {}
                            }
                            rs, meta = rule_score(applicant)
                            ms, mmeta = ml_score(applicant)
                            w_rule = st.sidebar.slider("Ensemble: Rule weight", 0.0, 1.0, 0.5, step=0.05)
                            final_score = w_rule*rs + (1-w_rule)*ms
                            recommendation = "Approve" if final_score < 30 else "Refer to Senior Underwriter" if final_score < 70 else "Decline"
                            display_result = {
                                'final_risk_score': round(final_score/10.0,2),
                                'recommendation': recommendation,
                                'justification': f"Rule score: {rs:.1f}, ML score: {ms:.1f}. Weighted ensemble used."
                            }
                            st.session_state.current_result = display_result
                            st.session_state.current_specialists = {
                                'claims': type("S", (), {"risk_score": int(meta.get('num_claims',0)), "summary":"Fallback claims summary"}),
                                'profile': type("S", (), {"risk_score": int((100-rs)/10 if rs else 5), "summary":"Fallback profile"}),
                                'external': type("S", (), {"risk_score":5, "summary":"Fallback external"})
                            }
                            append_audit(applicant['applicant_id'], {'risk_score': final_score, 'recommendation': recommendation, 'provenance': 'fallback'})
                    except Exception as e:
                        st.error(f"Processing error: {e}")

    with col_output:
        st.subheader("2. Analysis & Decision")
        if st.session_state.get("current_result"):
            res = st.session_state["current_result"]
            specs = st.session_state.get("current_specialists", {})
            # header
            if isinstance(res, dict):
                score_val = float(res.get('final_risk_score', 0))
                rec = res.get('recommendation', '')
                just = res.get('justification', '')
            else:
                score_val = float(getattr(res, "final_risk_score", 0))
                rec = getattr(res, "recommendation", "")
                just = getattr(res, "justification", "")
            # show gauge and recommendation
            col_a, col_b = st.columns([1,2])
            with col_a:
                st.plotly_chart(risk_gauge(score_val*10, title="Final Risk (0-10)"), use_container_width=True)
            with col_b:
                if rec == "Approve":
                    st.success(f"### ✅ {rec}")
                elif rec == "Decline":
                    st.error(f"### 🛑 {rec}")
                else:
                    st.warning(f"### ⚠️ {rec}")
                st.markdown(f"**Justification:** {just}")
            # radar
            st.divider()
            st.markdown("#### 📡 Risk Multi-Dimensional View")
            try:
                radar = get_radar_chart(specs)
                st.plotly_chart(radar, use_container_width=True)
            except Exception:
                st.info("Radar chart unavailable for the current result.")
            # tabs
            tab_logic, tab_letter = st.tabs(["🧠 Agent Reasoning", "✉️ Draft Decision Letter"])
            with tab_logic:
                st.info(f"**Final Justification:** {just}")
                with st.expander("See Specialist Breakdowns"):
                    try:
                        st.markdown(f"**Claims:** {specs['claims'].summary} (Score: {specs['claims'].risk_score})")
                        st.markdown(f"**Profile:** {specs['profile'].summary} (Score: {specs['profile'].risk_score})")
                        st.markdown(f"**External:** {specs['external'].summary} (Score: {specs['external'].risk_score})")
                    except Exception:
                        st.write("Specialist details not available for this result.")
            with tab_letter:
                if HAS_OLLAMA and letter_gen:
                    if st.button("Generate Formal Letter"):
                        with st.spinner("Drafting email..."):
                            try:
                                letter = letter_gen.invoke({
                                    "input_data": (raw_text or "")[:500],
                                    "decision": rec,
                                    "reasoning": just
                                })
                                st.text_area("Draft Email", value=getattr(letter, 'content', str(letter)), height=250)
                            except Exception as e:
                                st.error(f"Letter generation error: {e}")
                else:
                    st.info("Letter generation via Ollama unavailable in this environment.")

        else:
            st.info("Awaiting input data for analysis...")

# ---------- DEEP DIVE ----------
elif selected_page == "Deep Dive":
    st.title("🔬 Agent Deep Dive & Explainability")
    applicant_id_filter = st.text_input("Filter by applicant id (optional)", "")
    records = load_audit(applicant_id_filter.strip() or None)
    st.markdown(f"Found **{len(records)}** audit records.")
    if records:
        last = records[-1]
        st.write("Latest entry:", last['timestamp'], "-", last['applicant_id'])
        draw_timeline(records)
        st.markdown("### Explainability (Top Drivers)")
        if SHAP_AVAILABLE and ml_model:
            st.success("SHAP available — interactive simulation")
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.slider("Age", 18, 85, 40)
                credit = st.slider("Credit score", 300, 850, 650)
                num_claims = st.number_input("Number of prior claims", 0, 10, 0)
                total_amt = st.number_input("Total prior claim amount", 0, 50000, 0)
            with col_b:
                violations = st.slider("Driving violations (last 5y)", 0, 10, 0)
                vehicle_age = st.slider("Vehicle age", 0, 30, 5)
                loc = st.selectbox("Location risk", ["Low","Moderate","High"])
            sim_app = {'age': age, 'credit_score': credit, 'prior_claims':[{'amount': total_amt}]*int(num_claims), 'driving_violations_last_5y': int(violations), 'vehicle_age': vehicle_age, 'location_risk': loc}
            svals = shap_explain(sim_app)
            if svals:
                st.table(pd.DataFrame(svals))
            else:
                st.warning("SHAP explainability not available for this input.")
        else:
            st.warning("SHAP or ML model not available — showing rule-based breakdown option.")
            if st.button("Show rule breakdown for latest record"):
                st.json(last)

# ---------- LIVE CO-PILOT ----------
elif selected_page == "Live Co-Pilot":
    st.title("💬 Live Co-Pilot")
    st.markdown("Paste unstructured text (email/notes) or use the chat UI to analyze risk quickly.")
    tab_chat, tab_report = st.tabs(["Chat Interface", "Latest Report"])
    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role":"assistant","content":"Hello! Paste applicant data or a paragraph and get a quick risk estimate."}]
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if prompt := st.chat_input("Paste applicant data here..."):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        if proc_chain and synth_chain:
                            specialists_out = proc_chain.invoke({"input_data": prompt, "guidelines": ""})
                            final_out = synth_chain.invoke({
                                "claims": specialists_out['claims'],
                                "profile": specialists_out['profile'],
                                "external": specialists_out['external'],
                                "guidelines": ""
                            })
                            st.markdown(f"**LLM Result** — Score: {final_out.final_risk_score:.2f}/10 — {final_out.recommendation}")
                            st.markdown(final_out.justification)
                            append_audit("LIVE_"+str(int(time.time())), {'final_risk_score': final_out.final_risk_score*10, 'recommendation': final_out.recommendation, 'provenance':'llm'})
                        else:
                            cs = re.search(r"credit[:\\s]+(\\d{3})", prompt, re.IGNORECASE)
                            credit = int(cs.group(1)) if cs else 650
                            applicant = {'age':40,'credit_score':credit,'prior_claims':[],'driving_violations_last_5y':0,'occupation':'Unknown','vehicle_age':5,'location_risk':'Moderate'}
                            rs, meta = rule_score(applicant)
                            st.markdown(f"**Rule Fallback** — Score: {rs:.1f}/100 (approx {rs/10:.1f}/10)")
                            append_audit("LIVE_"+str(int(time.time())), {'risk_score': rs, 'recommendation': 'Auto-Review' if rs>30 else 'Approve', 'provenance':'rule'})
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        append_audit("LIVE_ERR_"+str(int(time.time())), {'risk_score':0,'recommendation':'Error','provenance':'error'})

    with tab_report:
        st.subheader("Latest Full Report")
        records = load_audit(None)
        if records:
            st.json(records[-1])
        else:
            st.info("No reports yet. Try the Chat Interface or PDF Processor.")

# ---------- BATCH SCORING ----------
elif selected_page == "Batch Scoring":
    st.title("📤 Batch Scoring & Export")
    st.markdown("Upload a CSV of applicants (columns: applicant_id, name, age, occupation, credit_score, annual_income, prior_claims_count, prior_claims_total, driving_violations_last_5y, location_risk).")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
    ensemble_weight = st.slider("Ensemble Weight: Rule vs ML (rule weight)", 0.0, 1.0, 0.5, step=0.05)
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview:", df.head())
        if st.button("Run batch scoring"):
            results = []
            progress = st.progress(0)
            total = len(df)
            for i, row in df.iterrows():
                applicant = {
                    'applicant_id': row.get('applicant_id', f"batch_{i}"),
                    'name': row.get('name', 'Unknown'),
                    'age': int(row.get('age', 40) or 40),
                    'occupation': row.get('occupation', 'Unknown'),
                    'annual_income': float(row.get('annual_income', 0) or 0),
                    'credit_score': int(row.get('credit_score', 650) or 650),
                    'prior_claims': [{'amount': row.get('prior_claims_total', 0)}]*int(row.get('prior_claims_count', 0) or 0),
                    'policy_type': row.get('policy_type', 'Auto'),
                    'vehicle_age': int(row.get('vehicle_age', 0) or 0),
                    'driving_violations_last_5y': int(row.get('driving_violations_last_5y', 0) or 0),
                    'location_risk': row.get('location_risk', 'Moderate'),
                    'external_reports': {}
                }
                rs, _ = rule_score(applicant)
                ms, _ = ml_score(applicant)
                final_score = ensemble_weight*rs + (1-ensemble_weight)*ms
                rec = "Approve" if final_score < 30 else "Refer to Senior Underwriter" if final_score < 70 else "Decline"
                out = {'applicant_id': applicant['applicant_id'], 'risk_score': final_score, 'recommendation': rec}
                results.append(out)
                append_audit(applicant['applicant_id'], {'risk_score': final_score, 'recommendation': rec, 'provenance':'batch'})
                time.sleep(0.05)
                progress.progress((i+1)/total)
            result_df = pd.DataFrame(results)
            st.success("Batch scoring complete")
            st.download_button("Download results CSV", result_df.to_csv(index=False), file_name="batch_results.csv")
            st.dataframe(result_df)

# ---------- COUNTERFACTUALS ----------
elif selected_page == "Counterfactuals":
    st.title("🧭 Counterfactual Analyzer & Ensemble Tuner")
    col1, col2 = st.columns(2)
    with col1:
        credit = st.slider("Credit score", 300, 850, 640)
        age = st.slider("Age", 18, 85, 40)
        violations = st.slider("Driving violations (last 5y)", 0, 10, 1)
        num_claims = st.slider("Number of prior claims", 0, 8, 1)
    with col2:
        total_claim_amt = st.number_input("Total prior claims amount", 0, 100000, 5000)
        vehicle_age = st.slider("Vehicle age", 0, 30, 6)
        location = st.selectbox("Location risk", ["Low","Moderate","High"])
        occupation = st.text_input("Occupation", "Delivery driver")
    ensemble_weight = st.slider("Rule weight (for ensemble)", 0.0, 1.0, 0.5, step=0.05)
    applicant = {
        'applicant_id': f"CF_{int(time.time())}",
        'name': "Counterfactual User",
        'age': age,
        'occupation': occupation,
        'annual_income': 30000,
        'credit_score': credit,
        'prior_claims': [{'amount': total_claim_amt}]*int(num_claims),
        'policy_type': 'Auto',
        'vehicle_age': vehicle_age,
        'driving_violations_last_5y': violations,
        'location_risk': location,
        'external_reports': {}
    }
    rs, rmeta = rule_score(applicant)
    ms, mmeta = ml_score(applicant)
    final_score = ensemble_weight*rs + (1-ensemble_weight)*ms
    rec = "Approve" if final_score < 30 else "Refer to Senior Underwriter" if final_score < 70 else "Decline"
    st.metric("Rule score (0-100)", f"{rs:.1f}")
    st.metric("ML score (0-100)", f"{ms:.1f}")
    st.metric("Final score (0-100)", f"{final_score:.1f}")
    st.markdown(f"**Recommendation:** {rec}")
    st.plotly_chart(risk_gauge(final_score, title="Counterfactual Final Risk (0-10 display)"), use_container_width=True)
    st.subheader("Feature Contributions")
    if SHAP_AVAILABLE and ml_model:
        svals = shap_explain(applicant)
        if svals:
            st.table(pd.DataFrame(svals))
    else:
        contribs = []
        contribs.append({'feature':'Credit score', 'impact': rmeta.get('raw_points',0) * (12 if applicant['credit_score']<650 else 0)})
        contribs.append({'feature':'Num claims', 'impact': rmeta.get('num_claims',0)*8})
        contribs.append({'feature':'Driving violations', 'impact': applicant['driving_violations_last_5y']*6})
        st.table(pd.DataFrame(contribs))
    if st.button("Export counterfactual as JSON"):
        st.download_button("Download JSON", json.dumps(applicant, indent=2), file_name=f"counterfactual_{int(time.time())}.json")

# ---------- SETTINGS ----------
elif selected_page == "Settings":
    st.title("⚙️ Configuration")
    st.markdown("Configure agent parameters and model sensitivity.")
    thresh = st.slider("Risk Threshold (Auto-Decline)", 0, 10, 8)
    debug = st.checkbox("Enable Debug Mode", value=False)
    st.markdown("Ollama available: " + ("Yes" if HAS_OLLAMA else "No"))
    st.markdown("SHAP available: " + ("Yes" if SHAP_AVAILABLE and ml_model else "No"))

# Footer / audit viewer
st.markdown("---")
st.markdown("**Audit & Tips**")
if st.checkbox("Show recent audit records"):
    recs = load_audit(None)[-20:]
    for r in recs[::-1]:
        st.write(r)
st.caption("Merged Workbench+ — highlights: LLM specialists (if available), Counterfactuals, TF-IDF RAG, Batch Scoring, Ensemble tuning, SHAP explainability (if available).")
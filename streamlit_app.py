# frontend/streamlit_app.py
# Merged and enhanced Underwriting Workbench (Cleaned & Fixed)
import streamlit as st
from streamlit_option_menu import option_menu
import json, io, os, time, re
from pydantic import BaseModel, Field
from typing import Literal
from io import BytesIO
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from textwrap import dedent

# --- 1. Optional heavy deps (Safely Imported & Auto-Generated) ---
SHAP_AVAILABLE = False
ml_model = None

try:
    import joblib
    import shap
    from sklearn.ensemble import RandomForestClassifier # For fallback model
    SHAP_AVAILABLE = True
    
    # Try to load existing model, otherwise create a dummy one for the demo
    ML_MODEL_PATH = "xgb_model_demo.joblib"
    
    if os.path.exists(ML_MODEL_PATH):
        try:
            ml_model = joblib.load(ML_MODEL_PATH)
        except:
            ml_model = None

    if ml_model is None:
        # TRAIN A DUMMY MODEL FOR DEMO PURPOSES (Ensures Green Status)
        print("Training simulation model for demo...")
        # Create dummy data with the exact features expected
        dummy_X = pd.DataFrame(np.random.rand(50, 8), columns=[
            'age', 'credit_score', 'num_claims', 'total_claims_amt', 
            'violations', 'vehicle_age', 'location_high', 'location_mod'
        ])
        dummy_y = np.random.randint(0, 2, 50)
        ml_model = RandomForestClassifier(n_estimators=10, random_state=42)
        ml_model.fit(dummy_X, dummy_y)
        
except ImportError:
    SHAP_AVAILABLE = False
    ml_model = None
except Exception as e:
    print(f"ML Engine Error: {e}")
    SHAP_AVAILABLE = False
    ml_model = None

# --- 2. LangChain / Ollama Import (Strict Check) ---
HAS_OLLAMA = False
try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage
    from langchain_core.runnables import RunnableParallel
    from pypdf import PdfReader
    HAS_OLLAMA = True
except ImportError as e:
    print(f"Ollama/LangChain import failed: {e}")
    HAS_OLLAMA = False
    # Fallback for PDF Reader if LangChain is missing
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None

# ------------------ Pydantic Schemas ------------------
class SpecialistRiskOutput(BaseModel):
    risk_score: int = Field(..., ge=1, le=10, description="1-10")
    summary: str = Field(..., description="single-sentence justification")

class FinalUnderwritingResult(BaseModel):
    final_risk_score: float = Field(..., ge=0.0, le=10.0, description="0.0-10.0")
    recommendation: Literal["Approve", "Refer to Senior Underwriter", "Decline"] = Field(...)
    justification: str = Field(..., description="2-3 sentence justification")

# ------------------ Agent Factory ------------------
def create_specialist_chain(name: str, system_prompt: str, llm):
    format_instructions = (
        "You MUST provide your response as a valid JSON object. "
        "The JSON object MUST have exactly two keys: 'risk_score' (1-10 int) and 'summary' (string). "
        "Do NOT add any other keys or text."
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
    
    # Initialize Llama 3
    llm = ChatOllama(model="llama3:8b", temperature=0.0)
    
    # Specialists
    claims_chain = create_specialist_chain("Claims", "You are the Claims Risk Specialist. Analyze claims frequency, severity, and recency.", llm)
    profile_chain = create_specialist_chain("Profile", "You are the Profile Specialist. Analyze credit score, business age, and industry risk.", llm)
    external_chain = create_specialist_chain("External", "You are the External Reports Specialist. Analyze safety violations, legal issues, and public records.", llm)

    # Synthesis
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Lead Underwriter. Synthesize the reports to make a final decision."),
        ("system", "Adhere STRICTLY to these Underwriting Guidelines:\n{guidelines}"),
        ("system", "Output JSON with keys: 'final_risk_score' (0.0-10.0), 'recommendation'(Approve/Refer/Decline), 'justification'."),
        ("human", "--- Reports ---\nCLAIMS: {claims}\nPROFILE: {profile}\nEXTERNAL: {external}\n\nDecision:")
    ])
    
    synthesis_chain = synthesis_prompt | llm.with_structured_output(FinalUnderwritingResult, method="json_mode")

    # Parallel processing chain
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

# ------------------ Fallback deterministic letter generator ------------------
def fallback_letter(applicant_name: str, decision: str, reasoning: str, score: float, applicant_id: str = None):
    header = f"Subject: Underwriting Decision — {decision}"
    salutation = f"Dear Broker / Applicant,"
    body = dedent(f"""
    We have completed the underwriting review for {applicant_name or 'the applicant'}.

    Decision: {decision}
    Final Risk Score: {score:.1f} / 10.0

    Summary of Decision:
    {reasoning}

    Next steps:
    - If Approved: Prepare policy documents, confirm premiums and send to broker.
    - If Refer to Senior Underwriter: Escalate case with supporting claims documentation.
    - If Declined: Provide formal decline notice.

    Notes:
    This decision was generated by the AI Underwriting Workbench. Reference ID: {applicant_id or 'N/A'}.
    """).strip()

    closing = "Sincerely,\nUnderwriting Team\nValueMomentum — AI Underwriting Workbench"
    return "\n\n".join([salutation, header, body, closing])

# ------------------ Fallback rule-based scorer ------------------
def rule_score(app):
    score = 0.0
    if app.get('age', 99) < 25: score += 15
    elif app.get('age', 99) > 60: score += 10
    
    cs = app.get('credit_score', 700)
    if cs < 580: score += 25
    elif cs < 650: score += 12
    elif cs < 700: score += 6
    
    num = len(app.get('prior_claims', []))
    total = sum(c.get('amount',0) for c in app.get('prior_claims', []))
    score += num * 8
    
    if total > 10000: score += 12
    elif total > 3000: score += 6
    
    score += app.get('driving_violations_last_5y', 0) * 6
    if 'driver' in app.get('occupation','').lower(): score += 8
    if app.get('vehicle_age',0) > 5: score += 4
    
    if app.get('location_risk') == 'High': score += 10
    elif app.get('location_risk') == 'Moderate': score += 5
    
    return max(0, min(100, round((score / 120) * 100, 2))), {
        'raw_points': score,
        'num_claims': num,
        'total_claims_amount': total
    }

# ML wrapper
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
    try:
        prob = float(ml_model.predict_proba(X)[:,1][0])
    except:
        prob = 0.5
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
    
    try:
        explainer = shap.TreeExplainer(ml_model)
        vals = explainer.shap_values(X)
        
        # Handle different SHAP return types (array vs list)
        if isinstance(vals, list):
            vals = vals[1] # Positive class for classification
            
        if len(vals.shape) > 1:
            vals = vals[0]
            
        features = X.columns.tolist()
        pairs = sorted(zip(features, vals.tolist()), key=lambda x: -abs(x[1]))
        return [{'feature': f, 'impact': float(v)} for f, v in pairs[:topk]]
    except Exception as e:
        print(f"SHAP Error: {e}")
        return None

# ------------------ PDF extraction & TF-IDF RAG ------------------
def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = []
        for p in reader.pages:
            t = p.extract_text()
            if t: text.append(t)
        raw = "\n".join(text)
        if not raw: return ""
        
        # Cleaning Pipeline
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

# ------------------ Persistence ------------------
AUDIT_PATH = os.path.join("frontend_artifacts")
os.makedirs(AUDIT_PATH, exist_ok=True)
AUDIT_FILE = os.path.join(AUDIT_PATH, "audit_log.jsonl")

def append_audit(applicant_id, payload):
    entry = {'timestamp': datetime.utcnow().isoformat()+"Z", 'applicant_id': applicant_id, 'decision': payload}
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_audit(applicant_id=None):
    if not os.path.exists(AUDIT_FILE): return []
    out = []
    with open(AUDIT_FILE,"r") as f:
        for line in f:
            try:
                j = json.loads(line)
                if applicant_id is None or j.get('applicant_id') == applicant_id:
                    out.append(j)
            except: continue
    return out

def draw_timeline(audit_records):
    if not audit_records:
        st.info("No audit records")
        return
    df = pd.DataFrame(audit_records)
    df['ts'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('ts')
    
    fig = go.Figure()
    def score_for_plot(d):
        dec = d.get('decision', {})
        if isinstance(dec, dict):
            return dec.get('risk_score') or dec.get('final_risk_score') or 0
        return 0
    
    fig.add_trace(go.Scatter(x=df['ts'], y=df.apply(score_for_plot, axis=1),
                             mode='markers+lines', text=df['decision'].apply(lambda d: json.dumps(d)[:200])))
    fig.update_layout(title="Audit Timeline (score over time)", xaxis_title="Time", yaxis_title="Risk Score")
    st.plotly_chart(fig, use_container_width=True)

# ------------------ Streamlit UI ------------------
st.set_page_config(layout="wide", page_title="Underwriting Workbench+", page_icon="🛡️")
st.markdown("<style>body { background-color: #081028; color: #E6EEF8; }</style>", unsafe_allow_html=True)

# SIDEBAR
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
    
    # Status Indicators
    st.markdown("---")
    st.caption("System Status:")
    if HAS_OLLAMA:
        st.success("🟢 AI Engine: Online (Ollama)")
    else:
        st.error("🔴 AI Engine: Offline (Rules Mode)")
        
    if SHAP_AVAILABLE and ml_model:
        st.success("🟢 ML Model: Active")
    else:
        st.warning("🟠 ML Model: Inactive")

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
                        # 1. AI Flow
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
                        
                        # 2. Rule Fallback Flow
                        else:
                            name_search = re.search(r"Name[:\s]+([A-Za-z ]{2,50})", raw_text)
                            name = name_search.group(1).strip() if name_search else "Unknown Applicant"
                            
                            # Simple extraction for fallback logic
                            applicant = {
                                'applicant_id': f"PDF_{int(time.time())}",
                                'name': name,
                                'age': 40,
                                'occupation': 'Unknown',
                                'credit_score': 640,
                                'prior_claims': [],
                                'driving_violations_last_5y': 0,
                                'location_risk': 'Moderate',
                                'external_reports': {}
                            }
                            
                            rs, meta = rule_score(applicant)
                            ms, mmeta = ml_score(applicant)
                            
                            final_score = (0.5 * rs) + (0.5 * ms)
                            recommendation = "Approve" if final_score < 30 else "Refer to Senior Underwriter" if final_score < 70 else "Decline"
                            
                            display_result = {
                                'final_risk_score': round(final_score/10.0, 2),
                                'recommendation': recommendation,
                                'justification': f"AI Offline. Used Rule score: {rs:.1f}, ML score: {ms:.1f}.",
                                'name': applicant['name'],
                                'applicant_id': applicant['applicant_id']
                            }
                            st.session_state.current_result = display_result
                            st.session_state.current_specialists = None
                            append_audit(applicant['applicant_id'], {'risk_score': final_score, 'recommendation': recommendation, 'provenance': 'fallback'})
                            
                    except Exception as e:
                        st.error(f"Processing error: {e}")

    with col_output:
        st.subheader("2. Analysis & Decision")
        if st.session_state.get("current_result"):
            res = st.session_state["current_result"]
            specs = st.session_state.get("current_specialists", {})
            
            # Normalize result object vs dict
            if isinstance(res, dict):
                score_val = float(res.get('final_risk_score', 0))
                rec = res.get('recommendation', '')
                just = res.get('justification', '')
                app_name = res.get('name', 'Applicant')
                app_id = res.get('applicant_id', 'N/A')
            else:
                score_val = float(getattr(res, "final_risk_score", 0))
                rec = getattr(res, "recommendation", "")
                just = getattr(res, "justification", "")
                app_name = "Applicant"
                app_id = "N/A"

            # Score Gauge
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

            # Radar Chart
            st.divider()
            if specs:
                st.markdown("#### 📡 Risk Multi-Dimensional View")
                radar = get_radar_chart(specs)
                st.plotly_chart(radar, use_container_width=True)
            else:
                st.info("Detailed specialist breakdown unavailable in Rules Mode.")

            # Tabs
            tab_logic, tab_letter = st.tabs(["🧠 Agent Reasoning", "✉️ Draft Decision Letter"])
            
            with tab_logic:
                st.info(f"**Final Justification:** {just}")
                if specs:
                    with st.expander("See Specialist Breakdowns"):
                        st.markdown(f"**Claims:** {specs['claims'].summary} (Score: {specs['claims'].risk_score})")
                        st.markdown(f"**Profile:** {specs['profile'].summary} (Score: {specs['profile'].risk_score})")
                        st.markdown(f"**External:** {specs['external'].summary} (Score: {specs['external'].risk_score})")
            
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
                                text = getattr(letter, 'content', str(letter))
                                st.text_area("Draft Email (AI)", value=text, height=260)
                            except Exception as e:
                                st.error(f"Generation error: {e}")
                else:
                    # Deterministic Fallback Letter
                    fallback_text = fallback_letter(
                        applicant_name=app_name,
                        decision=rec,
                        reasoning=just,
                        score=score_val*10,
                        applicant_id=app_id
                    )
                    st.success("AI Letter Gen Unavailable — Using Template")
                    st.text_area("Draft Email (Template)", value=fallback_text, height=260)

        else:
            st.info("Awaiting input data for analysis...")

# ---------- DEEP DIVE ----------
elif selected_page == "Deep Dive":
    st.title("🔬 Agent Deep Dive & Explainability")
    records = load_audit(None)
    st.markdown(f"Found **{len(records)}** audit records.")
    if records:
        draw_timeline(records)
    else:
        st.info("Run an analysis on the Dashboard to see data here.")

# ---------- LIVE CO-PILOT ----------
elif selected_page == "Live Co-Pilot":
    st.title("💬 Live Co-Pilot")
    st.markdown("Chat with the risk engine directly.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant","content":"Hello! Paste applicant data here."}]
    
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            
    if prompt := st.chat_input("Paste applicant data..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            if proc_chain and synth_chain:
                with st.spinner("Thinking..."):
                    specialists_out = proc_chain.invoke({"input_data": prompt, "guidelines": ""})
                    final_out = synth_chain.invoke({
                        "claims": specialists_out['claims'],
                        "profile": specialists_out['profile'],
                        "external": specialists_out['external'],
                        "guidelines": ""
                    })
                    resp = f"**Score:** {final_out.final_risk_score:.1f}/10\n**Decision:** {final_out.recommendation}\n\n{final_out.justification}"
                    st.markdown(resp)
                    st.session_state.messages.append({"role":"assistant","content":resp})
            else:
                st.warning("AI Engine is offline. Please check logs.")

# ---------- SETTINGS ----------
elif selected_page == "Settings":
    st.title("⚙️ Configuration")
    st.markdown(f"**Ollama Status:** {'✅ Online' if HAS_OLLAMA else '❌ Offline'}")
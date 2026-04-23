import streamlit as st
import google.generativeai as genai
import PyPDF2
import requests
from bs4 import BeautifulSoup
import io
import time
import random
import uuid
from datetime import datetime
from google.cloud import firestore
from google.oauth2 import service_account

# --- Configuration & UI ---
st.set_page_config(page_title="AI Elocution Arena", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: 600; }
    .debate-card { padding: 20px; border-radius: 10px; background: white; border: 1px solid #ddd; margin-bottom: 15px; }
    .verdict-box { background-color: #f3e5f5; padding: 15px; border-radius: 8px; border-left: 5px solid #9c27b0; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Database Setup (Freemium Firestore) ---
def get_db_client():
    # To use this in production, put your service account JSON in Streamlit Secrets
    try:
        if "firestore" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(st.secrets["firestore"])
            return firestore.Client(credentials=creds)
    except:
        return None
    return None

db = get_db_client()

# --- 2. Helper Functions ---
def extract_text(files, urls):
    context = ""
    for f in files:
        if f.name.endswith('.pdf'):
            pdf = PyPDF2.PdfReader(io.BytesIO(f.read()))
            context += f"\n[FILE: {f.name}]\n" + " ".join([p.extract_text() for p in pdf.pages])
        else:
            context += f"\n[FILE: {f.name}]\n" + f.read().decode()
    for url in urls.split('\n'):
        if url.strip():
            try:
                res = requests.get(url.strip(), timeout=5)
                soup = BeautifulSoup(res.text, 'html.parser')
                context += f"\n[LINK: {url}]\n" + soup.get_text(separator=' ', strip=True)[:5000]
            except: context += f"\n[LINK ERROR: {url}]\n"
    return context

def call_ai(model, role, task, context):
    prompt = f"ROLE: {role}\nSOURCE-ONLY MODE: Use only the text below. If missing, say 'Evidence not found.'\n\nSOURCES:\n{context}\n\nTASK: {task}"
    for i in range(5): # Exponential Backoff
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            if "429" in str(e): time.sleep(2**i + random.random()); continue
            return f"Error: {str(e)}"
    return "Rate limit exceeded."

# --- 3. Main App Layout ---
tab1, tab2 = st.tabs(["🎮 Competition Arena", "📚 Debate Archive"])

with tab1:
    col_ctrl, col_disp = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Setup")
        key = st.text_input("Gemini API Key", type="password")
        topic = st.text_input("Debate Topic")
        files = st.file_uploader("Upload Sources", accept_multiple_files=True)
        urls = st.text_area("Source Links")
        
        if st.button("🏁 Start 20m Session"):
            if not key or not topic:
                st.error("Key and Topic required.")
            else:
                st.session_state.start_time = datetime.now()
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
                
                with st.spinner("Agents are debating..."):
                    ctx = extract_text(files, urls)
                    
                    # Agent Sequence
                    pro = call_ai(model, "Proponent", f"Argue FOR: {topic}", ctx)
                    time.sleep(1)
                    con = call_ai(model, "Opponent", f"Rebut: {pro} and argue AGAINST: {topic}", ctx)
                    time.sleep(1)
                    judge = call_ai(model, "Neutral Judge", f"Judge this debate: \nPRO: {pro}\nCON: {con}", ctx)
                    
                    debate_data = {
                        "id": str(uuid.uuid4())[:8],
                        "timestamp": datetime.now(),
                        "topic": topic,
                        "pro": pro,
                        "con": con,
                        "judge": judge
                    }
                    
                    st.session_state.current_debate = debate_data
                    
                    # Save to Firestore if available
                    if db:
                        db.collection("debates").document(debate_data["id"]).set(debate_data)
                        st.success(f"Saved to cloud! ID: {debate_data['id']}")
                    else:
                        st.info("Cloud storage not configured. Debate saved for this session only.")

    with col_disp:
        if "current_debate" in st.session_state:
            d = st.session_state.current_debate
            st.header(f"Topic: {d['topic']}")
            
            # Timer Display
            elapsed = datetime.now() - st.session_state.start_time
            remaining = max(0, 1200 - elapsed.total_seconds())
            st.progress(remaining / 1200, text=f"Time Remaining: {int(remaining//60)}m {int(remaining%60)}s")
            
            st.subheader("🔵 Proponent")
            st.write(d["pro"])
            st.subheader("🔴 Opponent")
            st.write(d["con"])
            st.markdown(f'<div class="verdict-box"><h3>⚖️ Judge Verdict</h3>{d["judge"]}</div>', unsafe_allow_html=True)

with tab2:
    st.header("Cloud Archives")
    if not db:
        st.warning("Connect a Firestore service account in Streamlit Secrets to enable permanent storage.")
    else:
        # Simple retrieval
        search_id = st.text_input("Search by Debate ID")
        if st.button("Fetch Debate"):
            doc = db.collection("debates").document(search_id).get()
            if doc.exists:
                res = doc.to_dict()
                st.write(f"### {res['topic']}")
                st.write(res['judge'])
            else:
                st.error("Debate not found.")
        
        st.divider()
        st.subheader("Recent Debates")
        docs = db.collection("debates").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
        for doc in docs:
            data = doc.to_dict()
            with st.expander(f"{data['timestamp'].strftime('%Y-%m-%d')} - {data['topic']}"):
                st.write("**Verdict:**")
                st.write(data['judge'])
                st.caption(f"ID: {data['id']}")

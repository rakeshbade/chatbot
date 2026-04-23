import streamlit as st
import PyPDF2
import requests
from bs4 import BeautifulSoup
import io
import time
import random
import uuid
import json
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
    try:
        # Safer access to secrets to prevent crash if secrets.toml is missing
        if hasattr(st, "secrets") and "firestore" in st.secrets:
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

def call_openrouter(api_key, model_id, role, task, context):
    """Calls OpenRouter API with exponential backoff."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://localhost:8501", 
        "X-Title": "AI Elocution Arena",
        "Content-Type": "application/json"
    }
    
    prompt = f"ROLE: {role}\nSOURCE-ONLY MODE: Use only the text provided in SOURCES. If specific evidence is missing, state 'Evidence not found.'\n\nSOURCES:\n{context}\n\nTASK: {task}"
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    for i in range(5): 
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            elif response.status_code == 429:
                time.sleep(2**i + random.random())
                continue
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            time.sleep(2**i + random.random())
            if i == 4: return f"Connection Error: {str(e)}"
    
    return "Rate limit exceeded or API error."

# --- 3. Main App Layout ---
tab1, tab2 = st.tabs(["🎮 Competition Arena", "📚 Debate Archive"])

with tab1:
    col_ctrl, col_disp = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Setup")
        
        # Robust secrets check
        has_secrets = False
        or_key = None
        model_id = None
        
        try:
            if "openrouter_key" in st.secrets and "openrouter_model" in st.secrets:
                has_secrets = True
                or_key = st.secrets["openrouter_key"]
                model_id = st.secrets["openrouter_model"]
        except Exception:
            # st.secrets raises an error if the file is missing entirely
            has_secrets = False
        
        if not has_secrets:
            st.error("Missing OpenRouter configuration. Please create a `.streamlit/secrets.toml` file.")
            with st.expander("How to fix this?"):
                st.markdown("""
                Create a file at `.streamlit/secrets.toml` in your project root and add:
                ```toml
                openrouter_key = "your_key_here"
                openrouter_model = "google/gemini-2.0-flash-001"
                ```
                """)
        else:
            st.success(f"OpenRouter ready ({model_id})")

        topic = st.text_input("Debate Topic")
        files = st.file_uploader("Upload Sources", accept_multiple_files=True)
        urls = st.text_area("Source Links")
        
        if st.button("🏁 Start Debate Session", disabled=not has_secrets):
            if not topic:
                st.error("Debate Topic is required.")
            else:
                st.session_state.start_time = datetime.now()
                
                with st.spinner(f"Agents are debating using {model_id}..."):
                    ctx = extract_text(files, urls)
                    
                    # Agent Sequence
                    pro = call_openrouter(or_key, model_id, "Proponent", f"Argue FOR: {topic}", ctx)
                    time.sleep(0.5)
                    con = call_openrouter(or_key, model_id, "Opponent", f"Rebut: {pro} and argue AGAINST: {topic}", ctx)
                    time.sleep(0.5)
                    judge = call_openrouter(or_key, model_id, "Neutral Judge", f"Judge this debate objectively based on logical strength and usage of provided sources: \nPRO: {pro}\nCON: {con}", ctx)
                    
                    debate_data = {
                        "id": str(uuid.uuid4())[:8],
                        "timestamp": datetime.now(),
                        "topic": topic,
                        "model": model_id,
                        "pro": pro,
                        "con": con,
                        "judge": judge
                    }
                    
                    st.session_state.current_debate = debate_data
                    
                    # Save to Firestore if available
                    if db:
                        try:
                            db.collection("debates").document(debate_data["id"]).set(debate_data)
                            st.success(f"Saved to cloud! ID: {debate_data['id']}")
                        except Exception as e:
                            st.error(f"Firestore Error: {e}")
                    else:
                        st.info("Cloud storage not configured. View current session only.")

    with col_disp:
        if "current_debate" in st.session_state:
            d = st.session_state.current_debate
            st.header(f"Topic: {d['topic']}")
            st.caption(f"Model: {d.get('model', 'Unknown')}")
            
            # Timer Display (20m visual countdown)
            elapsed = datetime.now() - st.session_state.start_time
            remaining = max(0, 1200 - elapsed.total_seconds())
            st.progress(remaining / 1200, text=f"Session Remaining: {int(remaining//60)}m {int(remaining%60)}s")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔵 Proponent")
                st.write(d["pro"])
            with col2:
                st.subheader("🔴 Opponent")
                st.write(d["con"])
                
            st.markdown(f'<div class="verdict-box"><h3>⚖️ Judge Verdict</h3>{d["judge"]}</div>', unsafe_allow_html=True)

with tab2:
    st.header("Cloud Archives")
    if not db:
        st.warning("Connect a Firestore service account in Streamlit Secrets to enable permanent storage.")
    else:
        search_id = st.text_input("Search by Debate ID")
        if st.button("Fetch Debate"):
            doc = db.collection("debates").document(search_id).get()
            if doc.exists:
                res = doc.to_dict()
                st.write(f"### Topic: {res['topic']}")
                st.caption(f"Model used: {res.get('model', 'N/A')}")
                st.write("**Verdict:**")
                st.write(res['judge'])
                
                with st.expander("Show Full Debate History"):
                    st.write("**Pro:**", res['pro'])
                    st.write("**Con:**", res['con'])
            else:
                st.error("Debate not found.")
        
        st.divider()
        st.subheader("Recent Debates")
        try:
            docs = db.collection("debates").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
            for doc in docs:
                data = doc.to_dict()
                with st.expander(f"{data['timestamp'].strftime('%Y-%m-%d %H:%M')} - {data['topic']}"):
                    st.write(f"**Model:** {data.get('model', 'N/A')}")
                    st.write("**Verdict:**")
                    st.write(data['judge'])
                    st.caption(f"ID: {data['id']}")
        except Exception as e:
            st.error("Could not load recent debates.")
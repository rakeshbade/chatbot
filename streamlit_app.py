import streamlit as st
import PyPDF2
import requests
from bs4 import BeautifulSoup
import io
import time
import random
import uuid
import json
import threading
from datetime import datetime
from google.cloud import firestore
from google.oauth2 import service_account
from streamlit.runtime.scriptrunner import add_script_run_ctx

# --- Configuration & UI ---
st.set_page_config(page_title="AI Elocution Arena", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: 600; }
    .debate-card { padding: 20px; border-radius: 10px; background: white; border: 1px solid #ddd; margin-bottom: 15px; }
    .verdict-box { background-color: #f3e5f5; padding: 15px; border-radius: 8px; border-left: 5px solid #9c27b0; }
    .task-box { background-color: #e3f2fd; padding: 10px 15px; border-radius: 8px; border-left: 5px solid #2196f3; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "tasks" not in st.session_state:
    st.session_state.tasks = {}
if "completed_debates" not in st.session_state:
    st.session_state.completed_debates = []
if "fetch_result" not in st.session_state:
    st.session_state.fetch_result = None

# --- 1. Database Setup (Freemium Firestore) ---
def get_db_client():
    try:
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

def stream_openrouter(api_key, model_id, role, task_prompt, context, task_dict, key_to_update):
    """Calls OpenRouter API with streaming to update the UI token-by-token."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://localhost:8501", 
        "X-Title": "AI Elocution Arena",
        "Content-Type": "application/json"
    }
    
    prompt = f"ROLE: {role}\nSOURCE-ONLY MODE: Use only the text provided in SOURCES. If specific evidence is missing, state 'Evidence not found.'\n\nSOURCES:\n{context}\n\nTASK: {task_prompt}"
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    # Pre-fill so UI immediately knows we are doing something
    task_dict[key_to_update] = "⏳ Reading sources & thinking..."
    full_text = ""

    for i in range(5): 
        try:
            # Increased timeout for large contexts (TTFT can be long)
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
            if response.status_code == 200:
                task_dict[key_to_update] = ""  # Clear the thinking message
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    chunk = data['choices'][0]['delta'].get('content', '')
                                    full_text += chunk
                                    # Append cursor for visual streaming feedback
                                    task_dict[key_to_update] = full_text + " ▌"  
                            except:
                                pass
                task_dict[key_to_update] = full_text  # Remove cursor when done
                return full_text
            elif response.status_code == 429:
                task_dict[key_to_update] = "⏳ Rate limited. Retrying..."
                time.sleep(2**i + random.random())
                continue
            else:
                err = f"API Error: {response.status_code} - {response.text}"
                task_dict[key_to_update] = err
                return err
        except Exception as e:
            task_dict[key_to_update] = f"⏳ Connection issue. Retrying... ({str(e)})"
            time.sleep(2**i + random.random())
            if i == 4: 
                err = f"Connection Error: {str(e)}"
                task_dict[key_to_update] = err
                return err
    
    err = "Rate limit exceeded or API error after retries."
    task_dict[key_to_update] = err
    return err

def run_debate_bg(task_id, topic, ctx, model_id, or_key, db_client):
    """Background thread function generating the debate while pushing streaming updates."""
    task = st.session_state.tasks[task_id]
    try:
        task["status"] = "Proponent is forming arguments..."
        pro = stream_openrouter(or_key, model_id, "Proponent", f"Argue FOR: {topic}", ctx, task, "pro")
        task["pro"] = pro # Ensure finalized text is set
        
        task["status"] = "Opponent is formulating rebuttal..."
        con = stream_openrouter(or_key, model_id, "Opponent", f"Rebut: {pro} and argue AGAINST: {topic}", ctx, task, "con")
        task["con"] = con
        
        task["status"] = "Judge is evaluating..."
        judge = stream_openrouter(or_key, model_id, "Neutral Judge", f"Judge this debate objectively based on logical strength and usage of provided sources: \nPRO: {pro}\nCON: {con}", ctx, task, "judge")
        task["judge"] = judge
        
        debate_data = {
            "id": task_id,
            "timestamp": datetime.now(),
            "topic": topic,
            "model": model_id,
            "pro": pro,
            "con": con,
            "judge": judge
        }
        
        st.session_state.completed_debates.insert(0, debate_data)
        task["status"] = "Completed"
        
        if db_client:
            try:
                db_client.collection("debates").document(task_id).set(debate_data)
            except Exception as e:
                print(f"Firestore save error: {e}")
                
    except Exception as e:
        task["status"] = f"Error: {e}"

# --- 3. Main App Layout ---
tab1, tab2 = st.tabs(["🎮 Competition Arena", "📚 Debate Archive"])

with tab1:
    col_ctrl, col_disp = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Setup")
        
        has_secrets = False
        or_key = None
        model_id = None
        
        try:
            if "openrouter_key" in st.secrets and "openrouter_model" in st.secrets:
                has_secrets = True
                or_key = st.secrets["openrouter_key"]
                model_id = st.secrets["openrouter_model"]
        except Exception:
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
        
        can_start = bool(topic.strip()) and has_secrets

        if st.button("🏁 Start Debate Session", disabled=not can_start):
            with st.spinner("Extracting sources..."):
                ctx = extract_text(files, urls)
                
            task_id = str(uuid.uuid4())[:8]
            st.session_state.tasks[task_id] = {
                "topic": topic,
                "status": "Starting agents...",
                "start_time": datetime.now()
            }
            
            thread = threading.Thread(
                target=run_debate_bg, 
                args=(task_id, topic, ctx, model_id, or_key, db)
            )
            add_script_run_ctx(thread)
            thread.start()
            
            st.rerun() 

    with col_disp:
        # Define live render function
        def render_live_debates():
            header_col, btn_col = st.columns([3, 1])
            with header_col:
                st.header("Live & Recent Debates")
            with btn_col:
                if st.button("🔄 Refresh Status"):
                    st.rerun()
                    
            for tid, task in list(st.session_state.tasks.items()):
                if task["status"] not in ["Completed", "Error"]:
                    elapsed = (datetime.now() - task["start_time"]).total_seconds()
                    
                    with st.expander(f"⏳ {task['topic']} - {task['status']} ({int(elapsed)}s)", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🔵 Proponent")
                            if "pro" in task:
                                st.write(task["pro"])
                            else:
                                st.caption("Waiting to begin...")
                                
                        with col2:
                            st.subheader("🔴 Opponent")
                            if "con" in task:
                                st.write(task["con"])
                            else:
                                st.caption("Waiting for Proponent...")
                                
                        if "judge" in task:
                            st.markdown(f'<div class="verdict-box"><h3>⚖️ Judge Verdict</h3>{task["judge"]}</div>', unsafe_allow_html=True)
                                
                elif task["status"].startswith("Error"):
                    st.error(f"❌ **{task['topic']}** - {task['status']}")
                    
            if len(st.session_state.completed_debates) == 0 and len(st.session_state.tasks) == 0:
                st.info("No debates yet. Start a session from the left panel!")

        # Use Streamlit fragments to auto-refresh the right panel without interrupting typing on the left
        has_active_tasks = any(t["status"] not in ["Completed", "Error"] for t in st.session_state.tasks.values())
        
        if hasattr(st, "fragment") and has_active_tasks:
            @st.fragment(run_every=2)
            def live_auto_refresh_wrapper():
                render_live_debates()
                
            live_auto_refresh_wrapper()
        else:
            render_live_debates()

        # Display Completed Debates Outside the Live Render
        for d in st.session_state.completed_debates:
            with st.expander(f"✅ {d['topic']} (ID: {d['id']})", expanded=True):
                st.caption(f"Model: {d.get('model', 'Unknown')} | Time: {d['timestamp'].strftime('%H:%M:%S')}")
                
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
        
        if st.button("Fetch Debate", disabled=not search_id.strip()):
            with st.spinner("Fetching from database..."):
                doc = db.collection("debates").document(search_id).get()
                if doc.exists:
                    st.session_state.fetch_result = doc.to_dict()
                else:
                    st.session_state.fetch_result = "Not Found"
                    
        if st.session_state.fetch_result == "Not Found":
            st.error("Debate not found.")
        elif st.session_state.fetch_result:
            res = st.session_state.fetch_result
            st.write(f"### Topic: {res['topic']}")
            st.caption(f"Model used: {res.get('model', 'N/A')}")
            st.write("**Verdict:**")
            st.write(res['judge'])
            
            with st.expander("Show Full Debate History"):
                st.write("**Pro:**", res['pro'])
                st.write("**Con:**", res['con'])
        
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
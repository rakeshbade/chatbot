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
import re
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
    .verdict-box { background-color: #f3e5f5; padding: 15px; border-radius: 8px; border-left: 5px solid #9c27b0; margin-top: 20px;}
    .task-box { background-color: #e3f2fd; padding: 10px 15px; border-radius: 8px; border-left: 5px solid #2196f3; margin-bottom: 10px; }
    .turn-box { padding: 10px 0; }
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

def stream_openrouter(api_key, model_id, role, task_prompt, context, target_dict):
    """Calls OpenRouter API with highly robust streaming, retry, and keep-alive handling."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://localhost:8501", 
        "X-Title": "AI Elocution Arena",
        "Content-Type": "application/json"
    }
    
    if role == "Neutral Judge":
        prompt = f"ROLE: {role}\nINSTRUCTION: Before answering, ALWAYS wrap your detailed evaluation and thinking process in <think>...</think> tags.\n\nTASK: {task_prompt}\n\nYou must evaluate the full debate transcript against the provided reference SOURCES.\n\nREFERENCE SOURCES:\n{context}"
    else:
        prompt = f"ROLE: {role}\nINSTRUCTION: Before answering, ALWAYS wrap your detailed thinking process and strategy in <think>...</think> tags. Then provide your official argument.\nSOURCE-ONLY MODE: Use only the text provided in SOURCES.\n\nSOURCES:\n{context}\n\nTASK: {task_prompt}"
    
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "include_reasoning": True 
    }
    
    target_dict["text"] = "⏳ Reading sources & connecting..."
    target_dict["thinking"] = ""

    for i in range(5): 
        full_text = ""
        full_thinking = ""
        
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
            
            if response.status_code == 200:
                got_first_chunk = False
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8').strip()
                        
                        # Handle Keep-Alives used by long-thinking models to prevent drops
                        if line_str.startswith(":"):
                            if not got_first_chunk:
                                target_dict["text"] = "⏳ Model is reading & thinking (keep-alive received)..."
                            continue
                            
                        if line_str == "data: [DONE]":
                            continue

                        if line_str.startswith("data: "):
                            line_str = line_str[6:]

                        try:
                            data = json.loads(line_str)
                            
                            if 'error' in data:
                                raise ValueError(f"API Error: {data['error']}")
                                
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                chunk_text = ""
                                chunk_think = ""
                                
                                # Handle Streaming Delta
                                if 'delta' in choice:
                                    chunk_text = choice['delta'].get('content', '') or ''
                                    chunk_think = choice['delta'].get('reasoning', '') or ''
                                # Handle Non-Streaming Fallback (If free model ignores stream=True)
                                elif 'message' in choice:
                                    chunk_text = choice['message'].get('content', '') or ''
                                    chunk_think = choice['message'].get('reasoning', '') or ''

                                if chunk_think:
                                    full_thinking += chunk_think
                                    target_dict["thinking"] = full_thinking
                                    got_first_chunk = True

                                if chunk_text:
                                    full_text += chunk_text
                                    target_dict["text"] = full_text + " ▌"  
                                    got_first_chunk = True
                                    
                                if not got_first_chunk and not target_dict["text"].startswith("⏳"):
                                     target_dict["text"] = "⏳ Receiving data stream..."

                        except json.JSONDecodeError:
                            pass
                            
                # If API connection succeeded but yielded 0 text/thinking bytes (Free Model Glitch)
                if not full_text and not full_thinking:
                    raise ValueError("Empty 0-byte response received from model.")
                    
                target_dict["text"] = full_text  # Remove cursor upon completion
                return full_text
                
            elif response.status_code == 429:
                target_dict["text"] = f"⏳ Rate limited. Retrying {i+1}/5..."
                time.sleep(2**i + random.random())
                continue
            else:
                err = f"API Error: {response.status_code} - {response.text}"
                target_dict["text"] = err
                return err
                
        except Exception as e:
            target_dict["text"] = f"⏳ Connection issue. Retrying {i+1}/5... ({str(e)})"
            time.sleep(2**i + random.random())
            if i == 4: 
                err = f"Connection Failed: {str(e)}"
                target_dict["text"] = err
                return err
    
    err = "Rate limit exceeded or API error after 5 retries."
    target_dict["text"] = err
    return err

def run_debate_bg(task_id, topic, ctx, model_id, or_key, db_client):
    """Background thread running a 10-turn debate and pushing streaming updates."""
    task = st.session_state.tasks[task_id]
    task["turns"] = []
    
    try:
        # --- 10 Turn Conversation Loop ---
        for i in range(10):
            role = "Proponent" if i % 2 == 0 else "Opponent"
            task["status"] = f"Turn {i+1}/10: {role} is arguing..."
            
            turn_dict = {"role": role, "text": "", "thinking": ""}
            task["turns"].append(turn_dict)
            
            # Build clean history (stripping out previous <think> tags so models don't read opponent's minds)
            clean_history = []
            for t in task["turns"][:-1]:
                clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', t['text'], flags=re.DOTALL).strip()
                clean_history.append(f"{t['role']}: {clean_text}")
            history_text = "\n\n".join(clean_history)
            
            if i == 0:
                prompt = f"Make your opening argument FOR: {topic}"
            elif i == 1:
                prompt = f"Make your opening argument AGAINST: {topic}. Directly address the Proponent's points.\n\nDEBATE HISTORY:\n{history_text}"
            else:
                prompt = f"Provide your counter-argument. Address the opponent's latest points and strengthen your case.\n\nDEBATE HISTORY:\n{history_text}"
                
            stream_openrouter(or_key, model_id, role, prompt, ctx, turn_dict)
        
        # --- Judge Evaluation ---
        task["status"] = "Judge is evaluating the full transcript..."
        
        # Pass the full clean history to the judge
        clean_history = []
        for i, t in enumerate(task["turns"]):
            clean_text = re.sub(r'<think>.*?(?:</think>|$)', '', t['text'], flags=re.DOTALL).strip()
            clean_history.append(f"TURN {i+1} - {t['role']}: {clean_text}")
        full_transcript = "\n\n".join(clean_history)
        
        judge_dict = {"role": "Neutral Judge", "text": "", "thinking": ""}
        task["judge_data"] = judge_dict
        
        judge_prompt = f"Judge this 10-turn debate objectively based on logical strength, effective rebuttals, and accuracy against provided sources:\n\n{full_transcript}"
        stream_openrouter(or_key, model_id, "Neutral Judge", judge_prompt, ctx, judge_dict)
        
        # --- Save ---
        debate_data = {
            "id": task_id,
            "timestamp": datetime.now(),
            "topic": topic,
            "model": model_id,
            "turns": task["turns"],
            "judge_data": task["judge_data"]
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

# --- Rendering Helper ---
def render_turn(turn, is_live=False):
    """Renders a single agent's turn, separating thinking from the final response."""
    if not turn: return
    
    text = turn.get("text", "")
    thinking = turn.get("thinking", "")
    
    # Render loading placeholders early
    if text.startswith("⏳"):
        st.caption(text)
        if thinking:
            with st.expander("🧠 Internal Thinking Process", expanded=is_live):
                st.markdown(thinking)
        return
        
    if not text and not thinking:
        st.caption("⏳ Waiting for model to begin generating...")
        return

    # 1. Native API Reasoning (e.g. DeepSeek R1 via Native Spec)
    if thinking:
        with st.expander("🧠 Internal Thinking Process", expanded=is_live):
            st.markdown(thinking)
            
    # 2. Extract inline <think> tags (if model embeds it in text content)
    think_match = re.search(r'<think>(.*?)(?:</think>|$)', text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1).strip()
        # Display inline reasoning if we aren't already displaying native reasoning
        if think_content and not thinking:
            with st.expander("🧠 Internal Thinking Process", expanded=is_live):
                st.markdown(think_content)
                
        # Remove the think block from final text cleanly
        text_without_think = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        if text_without_think:
            st.markdown(text_without_think)
        elif is_live and text.endswith("▌"):
            st.markdown("▌") # Make sure cursor shows even when only thinking has generated
    else:
        if text:
            st.markdown(text)

def render_debate(data, is_live_task=False):
    """Renders the full debate transcript, supporting legacy and new structures."""
    if "turns" in data:
        for i, turn in enumerate(data["turns"]):
            role_icon = "🔵" if turn["role"] == "Proponent" else "🔴"
            st.markdown(f"<div class='turn-box'><h4>{role_icon} Turn {i+1}: {turn['role']}</h4></div>", unsafe_allow_html=True)
            
            # Auto-expand the thinking process only if it's the actively generating turn
            is_active_turn = is_live_task and i == len(data["turns"]) - 1 and "judge_data" not in data
            render_turn(turn, is_live=is_active_turn)
            st.divider()
            
        if "judge_data" in data and data["judge_data"].get("text"):
            st.markdown('<div class="verdict-box"><h3>⚖️ Judge Verdict</h3>', unsafe_allow_html=True)
            render_turn(data["judge_data"], is_live=is_live_task)
            st.markdown('</div>', unsafe_allow_html=True)
            
    # Fallback for old debates stored in DB before the 10-turn update
    elif "pro" in data:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔵 Proponent")
            st.write(data["pro"])
        with col2:
            st.subheader("🔴 Opponent")
            st.write(data["con"])
        if data.get("judge"):
            st.markdown(f'<div class="verdict-box"><h3>⚖️ Judge Verdict</h3>{data["judge"]}</div>', unsafe_allow_html=True)

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

        if st.button("🏁 Start 10-Turn Debate Session", disabled=not can_start):
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
                        render_debate(task, is_live_task=True)
                                
                elif task["status"].startswith("Error"):
                    st.error(f"❌ **{task['topic']}** - {task['status']}")
                    
            if len(st.session_state.completed_debates) == 0 and len(st.session_state.tasks) == 0:
                st.info("No debates yet. Start a session from the left panel!")

        # Fragment auto-refresh setup
        has_active_tasks = any(t["status"] not in ["Completed", "Error"] for t in st.session_state.tasks.values())
        
        if hasattr(st, "fragment") and has_active_tasks:
            @st.fragment(run_every=2)
            def live_auto_refresh_wrapper():
                render_live_debates()
                
            live_auto_refresh_wrapper()
        else:
            render_live_debates()

        for d in st.session_state.completed_debates:
            with st.expander(f"✅ {d['topic']} (ID: {d['id']})", expanded=True):
                st.caption(f"Model: {d.get('model', 'Unknown')} | Time: {d['timestamp'].strftime('%H:%M:%S')}")
                render_debate(d, is_live_task=False)

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
            
            with st.expander("Show Full Debate History", expanded=True):
                render_debate(res, is_live_task=False)
        
        st.divider()
        st.subheader("Recent Debates")
        try:
            docs = db.collection("debates").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
            for doc in docs:
                data = doc.to_dict()
                with st.expander(f"{data['timestamp'].strftime('%Y-%m-%d %H:%M')} - {data['topic']}"):
                    st.write(f"**Model:** {data.get('model', 'N/A')}")
                    st.caption(f"ID: {data['id']}")
                    if "judge_data" in data and data["judge_data"].get("text"):
                        st.write("**Verdict:**")
                        st.write(re.sub(r'<think>.*?</think>', '', data['judge_data']['text'], flags=re.DOTALL).strip())
                    elif "judge" in data:
                        st.write("**Verdict:**")
                        st.write(data['judge'])
        except Exception as e:
            st.error("Could not load recent debates.")
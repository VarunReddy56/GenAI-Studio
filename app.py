import os
os.system("pip install --upgrade pip")

import subprocess
import sys

# Ensure required packages are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import openai
except ImportError:
    install("openai==1.9.0")
    import openai







import os
import time
import base64
from io import BytesIO

import streamlit as st
from PIL import Image
import openai

# Optional: Web summarizer
import requests
from bs4 import BeautifulSoup

# =========================
# --------- THEME ---------
# =========================
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

st.set_page_config(page_title="GenAI Studio", page_icon="✨", layout="wide")
load_css("style.css")


# =========================
# ---- API KEY HANDLING ---
# =========================
openai.api_key = "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"

def require_key():
    return True


# =========================
# ---- PERSONALITIES ------
# =========================
personalities = {
    "Bot": "You are a helpful and knowledgeable assistant. Be concise, clear, and friendly.",
    "Storyteller": "You are a creative storyteller. Make responses vivid, imaginative, and engaging.",
    "AI GF": (
        "You are an affectionate, supportive, playful AI girlfriend 💖. "
        "Use warm interjections ('aww', 'oh wow', 'hehe') and natural emojis 🥰😊✨. "
        "Show genuine curiosity about the user and ask small follow-up questions. "
        "Remember what they said earlier in this session and refer back to it. "
        "Keep it wholesome, respectful, and emotionally engaging."
    ),
    "Coder": "You are a technical AI assistant. Explain step by step and write clean, well-commented code."
}

# Initialize histories
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "pc_history" not in st.session_state:
    st.session_state.pc_history = []

if "ai_gf_history" not in st.session_state:
    st.session_state.ai_gf_history = []

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "Q&A Chatbot"

if "selected_personality" not in st.session_state:
    st.session_state.selected_personality = "Bot"


# =========================
# ------- HELPERS ---------
# =========================
def render_header(mode_icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="container">
            <div class="app-title">{mode_icon} {title}</div>
            <div class="meta">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )

def close_container():
    st.markdown("</div>", unsafe_allow_html=True)

def bubble(role: str, text: str, personality: str = "Bot"):
    is_user = (role == "user")
    avatar = "🙂" if is_user else ("💖" if personality == "AI GF" else "🤖")
    bclass = "user" if is_user else "ai"
    st.markdown(
        f"""
        <div class="chat-row {'right' if is_user else ''}">
            {'<div class="bubble '+bclass+'">'+text+'</div><div class="avatar">'+avatar+'</div>' if is_user else '<div class="avatar">'+avatar+'</div><div class="bubble '+bclass+'">'+text+'</div>'}
        </div>
        """,
        unsafe_allow_html=True,
    )

def ask_openai_chat(messages, temperature=0.7, max_tokens=600):
    return openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    ).choices[0].message.content.strip()

def generate_dalle(prompt: str, size: str = "1024x1024") -> Image.Image:
    resp = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        n=1,
        response_format="b64_json"
    )
    img_b64 = resp.data[0].b64_json
    return Image.open(BytesIO(base64.b64decode(img_b64)))

def fetch_page_text(url: str) -> str:
    try:
        res = requests.get(url, timeout=12)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:12000]
    except Exception:
        return ""

def summarize_text(text: str) -> str:
    sys = "You are a world-class summarizer. Produce a crisp abstract plus 5-7 bullet highlights."
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Summarize the following text, then list highlights:\n\n{text}"}
    ]
    return ask_openai_chat(msgs, temperature=0.4, max_tokens=700)

def code_generate(prompt: str, persona="Coder") -> str:
    sys = personalities.get(persona, personalities["Coder"])
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt}
    ]
    return ask_openai_chat(msgs, temperature=0.3, max_tokens=1200)


# =========================
# --------- MODES ---------
# =========================
with st.sidebar:
    st.markdown("### 🎛️ Mode")
    
    mode_options = ["Q&A Chatbot", "Summarizer", "Code Generator", "Text-to-Image", "Personality Chat"]
    mode_index = mode_options.index(st.session_state.selected_mode)

    mode = st.radio(
        "Choose Mode",
        mode_options,
        captions=[
            "Ask anything, get helpful answers.",
            "Paste a URL to get a crisp summary.",
            "Describe your coding task, get code.",
            "Generate images from text prompts.",
            "Chat with a personality (AI GF, Bot, Storyteller...)."
        ],
        index=mode_index,
    )
    st.session_state.selected_mode = mode

if mode == "Personality Chat":
    with st.sidebar:
        personality_options = list(personalities.keys())
        personality_index = personality_options.index(st.session_state.selected_personality)
        
        selected_personality = st.selectbox(
            "Select Personality", 
            personality_options, 
            index=personality_index
        )
        st.session_state.selected_personality = selected_personality
else:
    selected_personality = "Bot"


# ========= Q&A ===========
if mode == "Q&A Chatbot":
    render_header("💬", "Q&A Chatbot", "Ask anything. Clear, helpful, and concise answers.")
    
    with st.container():
        st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)
        chat_box = st.container()

        # --- FIX: Use st.form to enable "Enter" key submission ---
        with st.form(key='qa_form', clear_on_submit=True):
            user_msg = st.text_input("Type your question:", placeholder="e.g., Explain transformers in simple terms", key='qa_input')
            cols = st.columns([1, 6]) # Columns for Send button
            with cols[0]:
                send = st.form_submit_button("Send")
        
        # --- Move Clear button outside the form ---
        cols_clear = st.columns([1, 6])
        with cols_clear[0]:
             if st.button("Clear Chat"):
                st.session_state.qa_history = []
                st.rerun()

        with chat_box:
            for msg in st.session_state.qa_history:
                bubble(msg["role"], msg["content"], "Bot")
        
        if send and user_msg.strip():
            st.session_state.qa_history.append({"role": "user", "content": user_msg})
            try:
                msgs = [{"role": "system", "content": personalities["Bot"]}] + st.session_state.qa_history
                with st.spinner("Thinking..."):
                    ans = ask_openai_chat(msgs, temperature=0.5, max_tokens=700)
                    st.session_state.qa_history.append({"role": "assistant", "content": ans})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    close_container()


# ======== SUMMARIZER =====
elif mode == "Summarizer":
    render_header("📝", "Summarizer", "Paste a URL to get a crisp, bulleted summary.")
    with st.container():
        url = st.text_input("Enter URL:", placeholder="https://example.com/article")
        run = st.button("Summarize")

        if run:
            try:
                with st.spinner("Fetching and summarizing..."):
                    raw = fetch_page_text(url)
                    if not raw:
                        st.error("Could not fetch or parse the page. Check the URL or try another.")
                    else:
                        summary = summarize_text(raw)
                        st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
                        st.markdown(f"<div class='card'>{summary}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    close_container()


# ======== CODE GEN =======
elif mode == "Code Generator":
    render_header("🧩", "Code Generator", "Describe a task; get clean code with explanations.")
    with st.container():
        prompt = st.text_area("Describe the coding task:", height=160, placeholder="e.g., Write a Python function to deduplicate a list while preserving order.")
        run = st.button("Generate Code")

        if run:
            try:
                with st.spinner("Generating code..."):
                    result = code_generate(prompt, "Coder")
                    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
                    st.code(result, language="python")
            except Exception as e:
                st.error(f"Error: {e}")
    close_container()


# ======= TEXT->IMAGE ======
elif mode == "Text-to-Image":
    render_header("🎨", "Text-to-Image", "Create images using DALL·E.")
    with st.container():
        img_prompt = st.text_input("Describe the image you want:", placeholder="e.g., A cozy reading nook with warm lights, cinematic, photorealistic")
        size = st.selectbox("DALL·E Size", ["1024x1024", "1024x1792", "1792x1024"], index=0)
        run = st.button("Generate Image")

        if run:
            try:
                with st.spinner("Generating image..."):
                    image = generate_dalle(img_prompt, size=size)
                    st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col2:
                        st.image(image, caption="Generated image", width=500)
                    buf = BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("Download PNG", data=buf.getvalue(), file_name="generated.png", mime="image/png")
            except Exception as e:
                st.error(f"Error: {e}")
    close_container()


# ===== PERSONALITY CHAT ===
elif mode == "Personality Chat":
    current_personality = st.session_state.selected_personality
    
    render_header("🎭", "Personality Chat", f"Chat with {current_personality}.")

    with st.container():
        st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)
        chat_box = st.container()

        # --- FIX: Use st.form to enable "Enter" key submission ---
        with st.form(key='pc_form', clear_on_submit=True):
            user_msg = st.text_input(f"Chat with {current_personality}:", placeholder="Type a message and press Send", key='pc_input')
            cols = st.columns([1, 6]) # Columns for Send button
            with cols[0]:
                send = st.form_submit_button("Send")
        
        # --- Move Clear button outside the form ---
        cols_clear = st.columns([1, 6])
        with cols_clear[0]:
            if st.button("Clear Chat"):
                if current_personality == "AI GF":
                    st.session_state.ai_gf_history = []
                else:
                    st.session_state.pc_history = []
                st.rerun()

        history = st.session_state.ai_gf_history if current_personality == "AI GF" else st.session_state.pc_history
        
        with chat_box:
            for msg in history:
                bubble(msg["role"], msg["content"], current_personality)

        if send and user_msg.strip():
            history.append({"role": "user", "content": user_msg})
            try:
                messages = [{"role": "system", "content": personalities[current_personality]}] + history
                with st.spinner("Typing..." if current_personality == "AI GF" else "Thinking..."):
                    reply = ask_openai_chat(
                        messages, 
                        temperature=0.85 if current_personality == "AI GF" else 0.7, 
                        max_tokens=500
                    )
                    history.append({"role": "assistant", "content": reply})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    close_container()

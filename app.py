"""
Web UI for the research assistant, with chat history in the sidebar.
Run:  streamlit run app.py   (keep this file next to agent.py)
"""
import json
import uuid
from pathlib import Path

import streamlit as st

from agent import SYSTEM_PROMPT, research

# Set to True to keep history in a local file so it survives browser refreshes.
# Use only when running on your own computer. On Streamlit Cloud the file would be
# shared by all visitors and wiped on reboot, so leave it False there.
SAVE_TO_FILE = False
HISTORY_FILE = Path("chat_history.json")

st.set_page_config(page_title="Research Assistant", page_icon="🔎", layout="centered")


# ---------- History helpers ----------
def load_chats():
    if SAVE_TO_FILE and HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_chats():
    if SAVE_TO_FILE:
        HISTORY_FILE.write_text(
            json.dumps(st.session_state.chats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def new_chat():
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {
        "title": "New chat",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],  # what the model sees
        "display": [],  # what the page shows: [role, text]
    }
    st.session_state.current = cid


# ---------- State ----------
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
    st.session_state.current = None
    if st.session_state.chats:
        st.session_state.current = list(st.session_state.chats)[-1]
    else:
        new_chat()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Chats")
    if st.button("➕ New chat", use_container_width=True):
        current = st.session_state.chats.get(st.session_state.current)
        if not current or current["display"]:  # don't stack empty chats
            new_chat()
        st.rerun()

    st.divider()

    for cid in reversed(list(st.session_state.chats)):
        chat = st.session_state.chats[cid]
        is_current = cid == st.session_state.current
        col_open, col_del = st.columns([5, 1])
        if col_open.button(
            chat["title"],
            key=f"open_{cid}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            st.session_state.current = cid
            st.rerun()
        if col_del.button("🗑️", key=f"del_{cid}"):
            del st.session_state.chats[cid]
            if not st.session_state.chats:
                new_chat()
            elif st.session_state.current == cid:
                st.session_state.current = list(st.session_state.chats)[-1]
            save_chats()
            st.rerun()

# ---------- Main page ----------
st.title("🔎 Research Assistant")
st.caption("Searches the web, reads pages, and summarizes with sources.")

chat = st.session_state.chats[st.session_state.current]

for role, text in chat["display"]:
    with st.chat_message(role):
        st.markdown(text)

if question := st.chat_input("Ask a research question..."):
    if not chat["display"]:
        chat["title"] = question[:30] + ("…" if len(question) > 30 else "")

    chat["display"].append(["user", question])
    with st.chat_message("user"):
        st.markdown(question)
    chat["messages"].append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.status("Researching...", expanded=True) as status:

            def on_step(name, args):
                icon = "🔍" if name == "web_search" else "📄"
                st.write(f"{icon} `{name}`: {list(args.values())[0]}")

            try:
                answer = research(chat["messages"], on_step=on_step)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                answer = f"Error: {e}"
                status.update(label="Failed", state="error")
        st.markdown(answer)

    chat["display"].append(["assistant", answer])
    save_chats()
    st.rerun()  # refresh so the sidebar shows the new chat title

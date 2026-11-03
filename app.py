import streamlit as st

from agent import SYSTEM_PROMPT, research

st.set_page_config(page_title="Research Assistant", page_icon="🔎", layout="centered")
st.title("🔎 Research Assistant")
st.caption("Searches the web, reads pages, and summarizes with sources.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.chat = []

with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ New chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

if question := st.chat_input("Ask a research question..."):
    st.session_state.chat.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.status("Researching...", expanded=True) as status:

            def on_step(name, args):
                icon = "🔍" if name == "web_search" else "📄"
                st.write(f"{icon} `{name}`: {list(args.values())[0]}")

            try:
                answer = research(st.session_state.messages, on_step=on_step)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                answer = f"Error: {e}"
                status.update(label="Failed", state="error")
        st.markdown(answer)

    st.session_state.chat.append(("assistant", answer))
import uuid

import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from app import (
    get_pdf_text,
    get_text_chunks,
    get_vectorstore,
    get_conversation_agent,
    ask_question,
)

@st.cache_resource
def build_agent():
    # The agent and its checkpointer are shared across sessions;
    # conversation state is isolated per thread_id instead.
    return get_conversation_agent()

@st.cache_resource
def build_vectorstore(raw_text: str):
    text_chunks = get_text_chunks(raw_text)
    return get_vectorstore(text_chunks)

# UI: render conversation turns
def handle_userinput(user_question: str, agent, vectorstore, thread_id: str):
    answer = ask_question(agent, vectorstore, user_question, thread_id)

    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("bot", answer))

    for role, content in st.session_state.chat_history:
        template = user_template if role == "user" else bot_template
        st.write(template.replace("{{MSG}}", content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.vectorstore = build_vectorstore(raw_text)
                st.session_state.chat_history = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.success("Documents processed successfully. You can start asking questions.")

    if user_question and st.session_state.get("vectorstore") is not None:
        agent = build_agent()
        handle_userinput(user_question, agent, st.session_state.vectorstore, st.session_state.thread_id)

if __name__ == '__main__':
    main()

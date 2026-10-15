import html
import uuid

import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from app import load_pdfs, build_vectorstore, build_agent, ask_question

# UI: render one conversation turn
def render_message(role: str, content: str):
    template = user_template if role == "user" else bot_template
    # Escape the text so model output can't break the HTML, and keep its line breaks
    message = html.escape(content).replace("\n", "<br>")
    st.write(template.replace("{{MSG}}", message), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("agent", None)

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True
        )
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                try:
                    pages = load_pdfs(pdf_docs)
                    vectorstore = build_vectorstore(pages)
                    st.session_state.agent = build_agent(vectorstore)
                except ValueError as err:
                    st.error(str(err))
                else:
                    st.session_state.chat_history = []
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.success(
                        f"Processed {len(pages)} pages into {len(vectorstore.get()['ids'])} chunks. "
                        "You can start asking questions."
                    )

    # Conversation so far, oldest first, so the latest message is at the bottom
    for role, content in st.session_state.chat_history:
        render_message(role, content)

    # st.chat_input stays pinned to the bottom of the page and returns each question only once
    user_question = st.chat_input("Ask a question about your documents")
    if user_question:
        if st.session_state.agent is None:
            st.warning("Upload your PDFs and click 'Process' first.")
        else:
            render_message("user", user_question)
            with st.spinner("Thinking"):
                answer = ask_question(
                    st.session_state.agent,
                    user_question,
                    st.session_state.thread_id,
                )
            render_message("bot", answer)
            st.session_state.chat_history += [("user", user_question), ("bot", answer)]

if __name__ == '__main__':
    main()

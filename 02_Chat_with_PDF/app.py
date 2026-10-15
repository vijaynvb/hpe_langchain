import os
import tempfile
from dataclasses import dataclass

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import RemoveMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

# Ollama server URL - override with OLLAMA_BASE_URL env var
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# The new question plus the last 3 question/answer pairs. Ollama runs llama3.2 with a
# 4k-token context, so older turns are dropped to leave room for the document excerpts.
MAX_MESSAGES = 7

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about the \
user's uploaded PDF documents.

Rules:
- Answer questions about the documents only from the document excerpts below.
- Use the earlier conversation to understand follow-up questions and to answer questions \
about the conversation itself.
- If the excerpts do not contain the answer, say you don't know.
- If the user just greets you, greet them back and invite a question about the documents.
- Keep answers concise and mention the file and page you used.

Document excerpts:
{context}
"""


# Runtime context carries this session's vector store into the middleware
@dataclass
class RuntimeContext:
    vectorstore: FAISS


# Step 1 - Load: one Document per PDF page, with the file name and page number as metadata
def load_pdfs(pdf_files) -> list[Document]:
    """Load uploaded PDFs (objects with .name and .getvalue(), like Streamlit's UploadedFile)."""
    docs = []
    for pdf in pdf_files:
        # PyPDFLoader reads from a file path, so save the upload to a temporary file first
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf.getvalue())
        try:
            pages = PyPDFLoader(tmp.name).load()
        finally:
            os.remove(tmp.name)
        for page in pages:
            page.metadata["source"] = pdf.name
        docs.extend(pages)
    return docs


# Step 2 - Split: overlapping chunks, small enough to retrieve precisely
def split_documents(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text found in these PDFs. Scanned PDFs need OCR before they can be searched.")
    return chunks


# Step 3 - Embed and store: chunks go into a FAISS vector store
def build_vectorstore(chunks: list[Document]) -> FAISS:
    embeddings = OllamaEmbeddings(model="embeddinggemma:latest", base_url=OLLAMA_BASE_URL)
    return FAISS.from_documents(documents=chunks, embedding=embeddings)


def format_excerpts(docs: list[Document]) -> str:
    excerpts = []
    for doc in docs:
        page = doc.metadata.get("page_label", doc.metadata.get("page", 0) + 1)
        excerpts.append(f"[{doc.metadata.get('source', 'document')}, page {page}]\n{doc.page_content}")
    return "\n\n".join(excerpts)


# Step 4 - Retrieve: search the PDFs for every question and put the best passages in the
# system prompt. Small models often skip optional tool calls, so retrieval always runs here.
@dynamic_prompt
def prompt_with_excerpts(request: ModelRequest) -> str:
    vectorstore = request.runtime.context.vectorstore
    questions = [message.text for message in request.messages if message.type == "human"]
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(questions[-1])
    if len(questions) > 1:
        # Also search the recent questions together, so a follow-up like
        # "can I use it for homework?" still finds the passages the earlier answer used
        docs += vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(" ".join(questions[-3:]))
    unique_docs = list({doc.page_content: doc for doc in docs}.values())
    return SYSTEM_PROMPT_TEMPLATE.format(context=format_excerpts(unique_docs))


# Step 5 - Memory: drop older turns before each model call so the prompt fits the context window
@before_model
def keep_recent_messages(state: AgentState, runtime: Runtime) -> dict | None:
    messages = state["messages"]
    if len(messages) <= MAX_MESSAGES:
        return None
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[-MAX_MESSAGES:]]}


# Create a LangChain v1 agent: retrieval middleware plus conversation memory
def get_conversation_agent():
    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=0.5,
        base_url=OLLAMA_BASE_URL,
    )
    return create_agent(
        model=llm,
        tools=[],  # retrieval runs in the middleware on every question, so no tool is needed
        middleware=[keep_recent_messages, prompt_with_excerpts],
        context_schema=RuntimeContext,
        checkpointer=InMemorySaver(),
    )


# Ask the agent a question; the thread_id keeps each browser session's conversation separate
def ask_question(agent, vectorstore: FAISS, question: str, thread_id: str) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
        context=RuntimeContext(vectorstore=vectorstore),
    )
    return result["messages"][-1].content

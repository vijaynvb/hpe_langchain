import os
import tempfile
import uuid

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
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
# context chunk sfrom the vector db queeried with similarity -3 chunks

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


# Steps 2 and 3 - Split into overlapping chunks, then embed them into a Chroma vector store
# (in memory, no persist_directory)
def build_vectorstore(docs: list[Document]) -> Chroma:
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    if not chunks:
        raise ValueError("No text found in these PDFs. Scanned PDFs need OCR before they can be searched.")
    embeddings = OllamaEmbeddings(model="embeddinggemma:latest", base_url=OLLAMA_BASE_URL)
    # In-memory Chroma clients in one process share their collections by name, so give every
    # upload its own collection. Otherwise sessions (or a re-upload) would mix their PDFs.
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf-{uuid.uuid4().hex}",
    )


# Step 4 - Memory: drop older turns before each model call so the prompt fits the context window
@before_model
def keep_recent_messages(state: AgentState, runtime: Runtime) -> dict | None:
    messages = state["messages"]
    if len(messages) <= MAX_MESSAGES:
        return None
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[-MAX_MESSAGES:]]}


# Create a LangChain v1 agent for one vector store: retrieval middleware plus conversation memory
def build_agent(vectorstore: Chroma):
    # Step 5 - Retrieve: search the PDFs for every question and put the best passages in the
    # system prompt. Small models often skip optional tool calls, so retrieval always runs here.
    @dynamic_prompt
    def prompt_with_excerpts(request: ModelRequest) -> str:
        questions = [message.text for message in request.messages if message.type == "human"]
        queries = [(questions[-1], 3)]
        if len(questions) > 1:
            # Also search the recent questions together, so a follow-up like
            # "can I use it for homework?" still finds the passages the earlier answer used
            queries.append((" ".join(questions[-3:]), 2))
        docs = [doc for query, k in queries for doc in vectorstore.similarity_search(query, k=k)]
        unique_docs = list({doc.page_content: doc for doc in docs}.values())
        excerpts = "\n\n".join(
            f"[{doc.metadata['source']}, page {doc.metadata['page_label']}]\n{doc.page_content}"
            for doc in unique_docs
        )
        prompt = SYSTEM_PROMPT_TEMPLATE.format(context=excerpts)
        print(prompt)
        return prompt

    return create_agent(
        model=ChatOllama(model="llama3.2:latest", temperature=0.5, base_url=OLLAMA_BASE_URL),
        tools=[],  # retrieval runs in the middleware on every question, so no tool is needed
        middleware=[keep_recent_messages, prompt_with_excerpts],
        checkpointer=InMemorySaver(),
    )


# Ask the agent a question; the thread_id keeps each conversation separate
def ask_question(agent, question: str, thread_id: str) -> str:
    # Print the configuration data passing into the middleware/agent
    print("\n--- Middleware Input Config ---")
    print(f"Thread ID: {thread_id}")
    print(f"User Question: {question}")
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    
    # Print the raw result dictionary to see all metadata and history
    print("\n--- Middleware Output Data ---")
    print(f"Full Result Keys: {list(result.keys())}")
    print(f"Total Messages in History: {len(result.get('messages', []))}")
    print(f"Last Agent Response: {result['messages'][-1].content}\n")
    
    return result["messages"][-1].content


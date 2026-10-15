import os

from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import get_runtime
from dataclasses import dataclass

# Ollama server URL - override with OLLAMA_BASE_URL env var
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the \
user's uploaded documents.

Rules:
- Always call the `search_documents` tool to find relevant passages before answering.
- Answer only using information returned by the tool.
- If the tool does not return anything relevant, say you don't know.
- Keep answers concise and cite the document content you relied on.
"""


# Runtime context carries the per-session vector store into the tool call
@dataclass
class RuntimeContext:
    vectorstore: FAISS


# Backend: extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Backend: split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# Backend: build FAISS vector store
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="embeddinggemma:latest", base_url=OLLAMA_BASE_URL)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


@tool
def search_documents(query: str) -> str:
    """Search the uploaded documents for passages relevant to the query."""
    runtime = get_runtime(RuntimeContext)
    docs = runtime.context.vectorstore.as_retriever().invoke(query)
    if not docs:
        return "No relevant passages found."
    return "\n\n".join(doc.page_content for doc in docs)


# Backend: create a LangChain v1 agent with retrieval tool and conversation memory
def get_conversation_agent():
    llm = ChatOllama(
        model="llama3.2:latest",
        temperature=0.5,
        base_url=OLLAMA_BASE_URL,
    )
    agent = create_agent(
        model=llm,
        tools=[search_documents],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


# Backend: ask the agent a question, keeping history via the thread_id
def ask_question(agent, vectorstore, question: str, thread_id: str) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
        context=RuntimeContext(vectorstore=vectorstore),
    )
    return result["messages"][-1].content

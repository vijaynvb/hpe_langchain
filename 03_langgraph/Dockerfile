FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Ollama runs on the host, not in this container
    OLLAMA_BASE_URL=http://host.docker.internal:11434

WORKDIR /app

# ipython is needed for the IPython.display import; requests for draw_mermaid_png
RUN pip install --no-cache-dir \
    "langgraph>=1.0.0" \
    langchain-core \
    langchain-ollama \
    python-dotenv \
    ipython \
    requests

COPY myapp.py .

RUN useradd --create-home appuser
USER appuser

CMD ["python", "myapp.py"]

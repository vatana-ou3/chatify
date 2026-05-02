# PDF Chatify

A Streamlit chatbot that lets a user upload a PDF, generate an explanatory guide, and ask questions grounded in the document.

## Stack

- Streamlit for the UI
- LangChain for orchestration
- PyPDF2 for PDF text extraction
- FastEmbed for low-latency local CPU embeddings
- ChromaDB for local vector storage
- Ollama, OpenAI, or Hugging Face Hub for the chat model

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional API keys and app configuration can be provided through a `.env` file:

```bash
set OPENAI_API_KEY=your_openai_key
set HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
set LLM_PROVIDER=Ollama
set OLLAMA_MODEL=frob/qwen3.5-instruct:4b
set EMBEDDING_BACKEND=fastembed
set EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
set OPENAI_EMBEDDING_MODEL=text-embedding-3-small
set MAX_OUTPUT_TOKENS=1200
set RETRIEVAL_K=3
set MAX_CONTEXT_CHARS=5000
set CHUNK_SIZE=1800
set CHUNK_OVERLAP=150
set SUMMARY_CHUNKS=16
set SUMMARY_MAX_CHARS=24000
set EXHAUSTIVE_BATCH_CHARS=5000
```

For local Qwen-style models through Ollama:

```bash
ollama pull frob/qwen3.5-instruct:4b
```

If your Ollama model has a different name, set `OLLAMA_MODEL` in `.env`.

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal.

## Hosting

For a hosted version, use OpenAI for both the chat model and embeddings. Keep Chroma as temporary local storage on the app server; uploaded PDFs will be indexed again if the host restarts.

For Streamlit Community Cloud, push this repo to GitHub, deploy `app.py`, then add these secrets in the app settings:

```toml
OPENAI_API_KEY = "your_openai_key"
LLM_PROVIDER = "OpenAI"
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_BACKEND = "openai"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_OUTPUT_TOKENS = "800"
MAX_CONTEXT_CHARS = "4000"
EXHAUSTIVE_BATCH_CHARS = "4000"
SUMMARY_CHUNKS = "8"
SUMMARY_MAX_CHARS = "12000"
```

Do not upload `.env`, `.chroma/`, `.fastembed_cache/`, or `.venv/` to GitHub. They are already ignored by `.gitignore`.

## Notes

- Chroma stores local indexes in `.chroma/`.
- Hosted mode should use `EMBEDDING_BACKEND=openai` so the server does not need local embedding model files.
- Chat uses direct Chroma similarity search plus one streamed LLM answer for lower latency. Turn on **Search whole document** in the app for broad questions that need every chunk; it scans the PDF in context-window-sized batches and merges the notes into a final answer.
- The Quiz tab can generate QCM/multiple-choice, context/open, or mixed quizzes from the uploaded PDF. You can choose the number of questions and set easy, medium, hard, or mixed difficulty.
- Summaries are generated on demand so uploaded documents become usable faster, especially on mobile.
- `BAAI/bge-small-en-v1.5` through FastEmbed is the default because it is much faster on laptop CPUs. Use `BAAI/bge-m3` with `EMBEDDING_BACKEND=sentence-transformers` only when you need higher retrieval quality and can accept slower query embedding.
- For 30 to 90 page PDFs, indexing will take longer on first upload, but Chroma reuses the saved index on later runs for the same PDF, embedding model, and chunk settings.
- Scanned PDFs need OCR first because PyPDF2 only extracts embedded text.
- The generated explanation samples chunks from across the document so later pages are represented. For very large PDFs, increase `SUMMARY_CHUNKS` or move the explanation step to a background map-reduce job.

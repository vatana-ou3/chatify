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
set RETRIEVAL_K=3
set MAX_CONTEXT_CHARS=5000
set CHUNK_SIZE=1800
set CHUNK_OVERLAP=150
set SUMMARY_CHUNKS=16
set SUMMARY_MAX_CHARS=24000
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

## Notes

- Chroma stores local indexes in `.chroma/`.
- Chat uses direct Chroma similarity search plus one streamed LLM answer for lower latency.
- `BAAI/bge-small-en-v1.5` through FastEmbed is the default because it is much faster on laptop CPUs. Use `BAAI/bge-m3` with `EMBEDDING_BACKEND=sentence-transformers` only when you need higher retrieval quality and can accept slower query embedding.
- For 30 to 90 page PDFs, indexing will take longer on first upload, but Chroma reuses the saved index on later runs for the same PDF, embedding model, and chunk settings.
- Scanned PDFs need OCR first because PyPDF2 only extracts embedded text.
- The generated explanation samples chunks from across the document so later pages are represented. For very large PDFs, increase `SUMMARY_CHUNKS` or move the explanation step to a background map-reduce job.

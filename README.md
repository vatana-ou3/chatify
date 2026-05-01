# PDF Chatify

A Streamlit chatbot that lets a user upload a PDF, generate a summary, and ask questions grounded in the document.

## Stack

- Streamlit for the UI
- LangChain for orchestration
- PyPDF2 for PDF text extraction
- bge-m3 through Sentence Transformers for embeddings
- ChromaDB for local vector storage
- Ollama, OpenAI, or Hugging Face Hub for the chat model

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional API keys can be provided in the sidebar or as environment variables:

```bash
set OPENAI_API_KEY=your_openai_key
set HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
```

For local Qwen-style models through Ollama:

```bash
ollama pull qwen3:8b
```

If your Ollama model has a different name, enter that model name in the sidebar.

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal.

## Notes

- Chroma stores local indexes in `.chroma/`.
- Scanned PDFs need OCR first because PyPDF2 only extracts embedded text.
- The summary currently uses the first document chunks so it stays fast for a prototype. For very large PDFs, upgrade this to a map-reduce summary chain.

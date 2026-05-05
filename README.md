# Chatify

A Streamlit chatbot that lets a user add a PDF or YouTube video, automatically generate an explanatory guide, and ask questions grounded in the source.

## Stack

- Streamlit for the UI
- LangChain for orchestration
- PyPDF2 for PDF text extraction
- youtube-transcript-api for YouTube captions
- yt-dlp and faster-whisper for optional YouTube audio transcription fallback
- Sentence Transformers for local CPU embeddings
- ChromaDB for local vector storage
- Gemini, Ollama, OpenAI, or Hugging Face Hub for the chat model

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional API keys and app configuration can be provided through a `.env` file:

```bash
set OPENAI_API_KEY=your_openai_key
set GEMINI_API_KEY=your_gemini_key
set HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
set LLM_PROVIDER=Gemini
set GEMINI_MODEL=gemini-2.5-flash
set OLLAMA_MODEL=frob/qwen3.5-instruct:4b
set EMBEDDING_BACKEND=sentence-transformers
set EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
set OPENAI_EMBEDDING_MODEL=text-embedding-3-small
set MAX_OUTPUT_TOKENS=1200
set RETRIEVAL_K=3
set MAX_CONTEXT_CHARS=50000
set CHUNK_SIZE=1800
set CHUNK_OVERLAP=150
set SUMMARY_CHUNKS=0
set SUMMARY_MAX_CHARS=750000
set EXHAUSTIVE_BATCH_CHARS=5000
set YOUTUBE_TRANSCRIPT_LANGUAGES=en
set YOUTUBE_WHISPER_MODEL=base
set YOUTUBE_MAX_WHISPER_MINUTES=90
```

For Gemini, create an API key in Google AI Studio, put it in `GEMINI_API_KEY`, and keep `GEMINI_MODEL=gemini-2.5-flash`.

For local Qwen-style models through Ollama:

```bash
ollama pull frob/qwen3.5-instruct:4b
```

If your Ollama model has a different name, set `OLLAMA_MODEL` in `.env`.

For YouTube videos, Chatify first tries existing YouTube captions. If no captions are available and the video is within the configured duration cap, it automatically downloads audio with `yt-dlp` and transcribes it locally with `faster-whisper`. Whisper fallback requires `ffmpeg` to be installed and available on your PATH.

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal.

## Hosting

For a hosted version, use Gemini for the chat model and a local sentence-transformers model for embeddings. Keep Chroma as temporary local storage on the app server; uploaded sources will be indexed again if the host restarts.

For Streamlit Community Cloud, push this repo to GitHub, deploy `app.py`, then add these secrets in the app settings:

```toml
GEMINI_API_KEY = "your_gemini_key"
LLM_PROVIDER = "Gemini"
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_BACKEND = "sentence-transformers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_OUTPUT_TOKENS = "800"
MAX_CONTEXT_CHARS = "50000"
EXHAUSTIVE_BATCH_CHARS = "4000"
SUMMARY_CHUNKS = "0"
SUMMARY_MAX_CHARS = "750000"
```

Do not upload `.env`, `.chroma/`, `.fastembed_cache/`, or `.venv/` to GitHub. They are already ignored by `.gitignore`.

## Notes

- Chroma stores local indexes in `.chroma/`.
- Hosted mode can use `EMBEDDING_BACKEND=sentence-transformers`, but the first run may download the embedding model and use more RAM than API embeddings.
- Chat uses direct Chroma similarity search plus one streamed LLM answer for lower latency. Turn on **Search whole source** in the app for broad questions that need every chunk; it scans the source in context-window-sized batches and merges the notes into a final answer.
- The Quiz tab can generate QCM/multiple-choice, context/open, or mixed quizzes from the uploaded source. You can choose the number of questions and set easy, medium, hard, or mixed difficulty.
- Summaries are generated automatically after source ingestion and indexing.
- `SUMMARY_CHUNKS=0` makes summaries use every extracted chunk. `SUMMARY_MAX_CHARS=750000` fits much more of a PDF when using Gemini's long context; lower it only if hosting memory or latency becomes a problem.
- `sentence-transformers/all-MiniLM-L6-v2` is the default because it is small, free, and works well on laptop CPUs. Use `BAAI/bge-m3` only when you need higher retrieval quality and can accept slower query embedding.
- For 30 to 90 page PDFs, indexing will take longer on first upload, but Chroma reuses the saved index on later runs for the same PDF, embedding model, and chunk settings.
- Scanned PDFs need OCR first because PyPDF2 only extracts embedded text.
- YouTube caption fetching is fast and does not download media. Whisper fallback is capped by `YOUTUBE_MAX_WHISPER_MINUTES` to avoid accidentally processing very long videos.
- The generated explanation samples chunks from across the document so later pages are represented. For very large PDFs, increase `SUMMARY_CHUNKS` or move the explanation step to a background map-reduce job.

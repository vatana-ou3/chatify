import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate


APP_TITLE = "Chatify"
DEFAULT_EMBEDDING_BACKEND = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_MODEL = "frob/qwen3.5-instruct:4b"  # Change this to your Ollama model name if different
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_PROVIDER = "Gemini"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 1200
DEFAULT_RETRIEVAL_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 50000
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SUMMARY_CHUNKS = 0
DEFAULT_SUMMARY_MAX_CHARS = 750000
DEFAULT_EXHAUSTIVE_BATCH_CHARS = 5000
DEFAULT_QUIZ_QUESTION_COUNT = 5
DEFAULT_YOUTUBE_LANGUAGES = "en"
DEFAULT_YOUTUBE_WHISPER_MODEL = "base"
DEFAULT_YOUTUBE_MAX_WHISPER_MINUTES = 90
CHROMA_DIR = ".chroma"
FASTEMBED_CACHE_DIR = ".fastembed_cache"
EMBEDDING_MODEL_ALIASES = {
    "bge-m3": "BAAI/bge-m3",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/bge-m3": "BAAI/bge-m3",
    "openai-small": DEFAULT_OPENAI_EMBEDDING_MODEL,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourcePayload:
    file_hash: str
    source_name: str
    source_type: str
    text: str
    item_count: int
    url: str = ""
    duration_seconds: int | None = None
    transcript_source: str = ""


def get_secret(name: str) -> str | None:
    value = os.getenv(name) or os.getenv(name.lower())
    if value:
        return value

    try:
        value = st.secrets.get(name) or st.secrets.get(name.lower())
    except Exception:
        return None

    return str(value) if value else None


def get_config_value(name: str, default: str | None = None) -> str | None:
    return get_secret(name) or default


def init_page() -> None:
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, page_icon="CHAT", layout="wide")
    title_col, action_col = st.columns([1, 0.18])
    with title_col:
        st.title(APP_TITLE)
        st.caption("Add a PDF or YouTube video, get a concise summary, then ask questions grounded in the source.")
    with action_col:
        if st.button("New chat", use_container_width=True):
            reset_chat()
            st.rerun()


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.summary = ""
    st.session_state.summary_key = ""
    st.session_state.quiz = ""
    st.session_state.quiz_items = []
    st.session_state.quiz_key = ""
    st.session_state.quiz_batch_count = 0
    st.session_state.active_file_hash = ""
    st.session_state.active_index_key = ""
    st.session_state.vector_store = None
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    st.session_state.youtube_url = ""


def read_pdf(uploaded_file) -> SourcePayload:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"\n\n[Page {page_number}]\n{text}")
    finally:
        os.unlink(tmp_path)

    return SourcePayload(
        file_hash=file_hash,
        source_name=uploaded_file.name,
        source_type="pdf",
        text="\n".join(pages).strip(),
        item_count=len(reader.pages),
    )


def parse_youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.rstrip("/")

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}:
        if path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif path.startswith(("/shorts/", "/embed/", "/live/")):
            parts = parsed.path.strip("/").split("/")
            video_id = parts[1] if len(parts) > 1 else ""
        else:
            video_id = ""
    else:
        video_id = ""

    if not re.fullmatch(r"[\w-]{11}", video_id):
        raise ValueError("Please enter a valid YouTube video URL.")
    return video_id


def get_youtube_proxy_url() -> str | None:
    return (
        get_config_value("YOUTUBE_PROXY_URL")
        or get_config_value("YOUTUBE_HTTPS_PROXY")
        or get_config_value("HTTPS_PROXY")
        or get_config_value("https_proxy")
    )


def create_youtube_transcript_api():
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError("Install youtube-transcript-api to fetch YouTube captions.") from exc

    webshare_username = get_config_value("WEBSHARE_PROXY_USERNAME")
    webshare_password = get_config_value("WEBSHARE_PROXY_PASSWORD")
    if webshare_username and webshare_password:
        try:
            from youtube_transcript_api.proxies import WebshareProxyConfig
        except ImportError as exc:
            raise RuntimeError("Update youtube-transcript-api to use Webshare proxy support.") from exc

        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=webshare_username,
                proxy_password=webshare_password,
            )
        )

    proxy_url = get_youtube_proxy_url()
    if proxy_url:
        try:
            from youtube_transcript_api.proxies import GenericProxyConfig
        except ImportError as exc:
            raise RuntimeError("Update youtube-transcript-api to use generic proxy support.") from exc

        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=get_config_value("YOUTUBE_HTTP_PROXY") or proxy_url,
                https_url=proxy_url,
            )
        )

    return YouTubeTranscriptApi()


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def segment_value(segment, name: str, default=0):
    if isinstance(segment, dict):
        return segment.get(name, default)
    return getattr(segment, name, default)


def fetch_youtube_captions(video_id: str, languages: list[str]) -> tuple[str, int]:
    transcript = create_youtube_transcript_api().fetch(video_id, languages=languages)
    lines = []
    last_end = 0
    for segment in transcript:
        start = float(segment_value(segment, "start", 0) or 0)
        duration = float(segment_value(segment, "duration", 0) or 0)
        text = str(segment_value(segment, "text", "")).replace("\n", " ").strip()
        if not text:
            continue
        last_end = max(last_end, int(start + duration))
        lines.append(f"[{format_timestamp(start)}] {text}")

    return "\n".join(lines), last_end


def fetch_youtube_info(url: str) -> dict:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Install yt-dlp to inspect YouTube video metadata or use Whisper fallback.") from exc

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    proxy_url = get_youtube_proxy_url()
    if proxy_url:
        ydl_opts["proxy"] = proxy_url
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def download_youtube_audio(url: str, output_dir: str) -> str:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("Install yt-dlp to download YouTube audio for Whisper transcription.") from exc

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
    }
    proxy_url = get_youtube_proxy_url()
    if proxy_url:
        ydl_opts["proxy"] = proxy_url
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    audio_files = [
        os.path.join(output_dir, file_name)
        for file_name in os.listdir(output_dir)
        if file_name.lower().endswith((".mp3", ".m4a", ".webm", ".opus", ".wav"))
    ]
    if not audio_files:
        raise RuntimeError("Audio download finished, but no audio file was found. Make sure ffmpeg is installed.")
    return audio_files[0]


def transcribe_audio_with_whisper(audio_path: str, model_name: str) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper to transcribe videos without captions.") from exc

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, vad_filter=True)
    lines = []
    for segment in segments:
        text = segment.text.replace("\n", " ").strip()
        if text:
            lines.append(f"[{format_timestamp(segment.start)}] {text}")
    return "\n".join(lines)


def read_youtube(
    url: str,
    languages: list[str],
    allow_whisper: bool,
    whisper_model: str,
    max_whisper_minutes: int,
) -> SourcePayload:
    video_id = parse_youtube_video_id(url)
    source_url = f"https://www.youtube.com/watch?v={video_id}"
    info = None
    title = f"YouTube video {video_id}"
    duration = None

    try:
        info = fetch_youtube_info(source_url)
        title = info.get("title") or title
        duration = info.get("duration")
    except Exception as exc:
        logger.info("Could not fetch YouTube metadata before captions: %s", exc)

    try:
        text, caption_duration = fetch_youtube_captions(video_id, languages)
        if text.strip():
            duration = duration or caption_duration
            file_hash = hashlib.sha256(f"youtube:{video_id}:{text}".encode("utf-8")).hexdigest()[:16]
            return SourcePayload(
                file_hash=file_hash,
                source_name=title,
                source_type="youtube",
                text=text,
                item_count=len(text.splitlines()),
                url=source_url,
                duration_seconds=int(duration) if duration else caption_duration,
                transcript_source="YouTube captions",
            )
    except Exception as exc:
        logger.info("YouTube captions unavailable for %s: %s", video_id, exc)
        if not allow_whisper:
            raise RuntimeError(
                "No usable YouTube captions were found. Enable Whisper fallback to transcribe the audio."
            ) from exc

    if info is None:
        info = fetch_youtube_info(source_url)
        title = info.get("title") or title
        duration = info.get("duration")

    max_seconds = max_whisper_minutes * 60
    if duration and duration > max_seconds:
        raise RuntimeError(
            f"This video is {round(duration / 60)} minutes long. Whisper fallback is capped at {max_whisper_minutes} minutes."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = download_youtube_audio(source_url, tmp_dir)
        text = transcribe_audio_with_whisper(audio_path, whisper_model)

    if not text.strip():
        raise RuntimeError("Whisper did not produce transcript text for this video.")

    file_hash = hashlib.sha256(f"youtube:{video_id}:whisper:{text}".encode("utf-8")).hexdigest()[:16]
    return SourcePayload(
        file_hash=file_hash,
        source_name=title,
        source_type="youtube",
        text=text,
        item_count=len(text.splitlines()),
        url=source_url,
        duration_seconds=int(duration) if duration else None,
        transcript_source=f"Whisper ({whisper_model})",
    )


def split_source(payload: SourcePayload, chunk_size: int, chunk_overlap: int) -> list[Document]:
    start = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(payload.text)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": payload.source_name,
                "source_type": payload.source_type,
                "url": payload.url,
                "file_hash": payload.file_hash,
                "chunk": index,
            },
        )
        for index, chunk in enumerate(chunks)
    ]
    logger.info(
        "Split %s %s items into %s chunks in %.2fs using chunk_size=%s chunk_overlap=%s",
        payload.item_count,
        payload.source_type,
        len(docs),
        time.perf_counter() - start,
        chunk_size,
        chunk_overlap,
    )
    return docs


@st.cache_resource(show_spinner=False)
def get_embeddings(backend: str, model_name: str):
    normalized_model_name = EMBEDDING_MODEL_ALIASES.get(model_name.strip(), model_name.strip())
    start = time.perf_counter()

    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=normalized_model_name or DEFAULT_OPENAI_EMBEDDING_MODEL,
            api_key=get_secret("OPENAI_API_KEY"),
        )
        logger.info("Loaded OpenAI embedding model %s in %.2fs", normalized_model_name, time.perf_counter() - start)
        return embeddings

    if backend == "fastembed":
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        embeddings = FastEmbedEmbeddings(
            model_name=normalized_model_name,
            cache_dir=FASTEMBED_CACHE_DIR,
            batch_size=256,
        )
        logger.info("Loaded FastEmbed model %s in %.2fs", normalized_model_name, time.perf_counter() - start)
        return embeddings

    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_kwargs = {"device": "cpu"}
    hf_token = get_secret("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        model_kwargs["token"] = hf_token

    embeddings = HuggingFaceEmbeddings(
        model_name=normalized_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("Loaded Sentence Transformers model %s in %.2fs", normalized_model_name, time.perf_counter() - start)
    return embeddings


def build_vector_store(source_type: str, file_hash: str, embedding_backend: str, embedding_model: str, docs: list[Document]):
    embeddings = get_embeddings(embedding_backend, embedding_model)
    embedding_hash = hashlib.sha256(f"{embedding_backend}:{embedding_model}".encode("utf-8")).hexdigest()[:8]
    collection_name = f"{source_type}_{file_hash}_{embedding_hash}"
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    if vector_store._collection.count() > 0:
        logger.info("Loaded existing Chroma collection %s with %s vectors", collection_name, vector_store._collection.count())
        return vector_store

    logger.info(
        "Creating Chroma collection %s with %s chunks using %s/%s",
        collection_name,
        len(docs),
        embedding_backend,
        embedding_model,
    )
    start = time.perf_counter()
    ids = [f"{source_type}-{file_hash}-{doc.metadata['chunk']}" for doc in docs]
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )
    logger.info("Created Chroma collection in %.2fs", time.perf_counter() - start)
    return vector_store


def get_llm(provider: str, model_name: str, temperature: float, api_key: str | None, max_output_tokens: int):
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_output_tokens,
            api_key=api_key or get_secret("OPENAI_API_KEY"),
        )

    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            google_api_key=api_key or get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY"),
        )

    if provider == "Hugging Face":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=temperature,
            max_new_tokens=max_output_tokens,
            huggingfacehub_api_token=api_key or get_secret("HUGGINGFACEHUB_API_TOKEN"),
        )
        return ChatHuggingFace(llm=endpoint)

    from langchain_ollama import ChatOllama

    return ChatOllama(model=model_name, temperature=temperature, num_predict=max_output_tokens)


SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are explaining a {source_type} for a reader. /no_think

Write a useful summary using only the source content below.
Do not include hidden reasoning, chain-of-thought, or <think> text.
Do not only say what the source is about. Explain the actual content so a reader can understand it without reading or watching the original.
Adapt the depth to the source length. For short videos, keep it compact and do not inflate simple material.

Use this structure:

## Quick summary
Explain the main message in plain language.

## Key details
List the important facts, examples, tools, commands, names, dates, numbers, warnings, or procedures that appear in the content.

## Takeaways
Explain what the reader should remember.

## Useful questions
Suggest a few questions the reader can ask next.

Source content:
{context}

Summary:"""
)


QA_PROMPT = PromptTemplate.from_template(
    """You answer questions about one uploaded source. /no_think
Use only the context below. If the answer is not in the source, say that the source does not contain enough information.
Do not include hidden reasoning, chain-of-thought, or <think> text.
Keep the answer concise and practical.

Context:
{context}

Question: {question}

Answer:"""
)


EXHAUSTIVE_BATCH_PROMPT = PromptTemplate.from_template(
    """You answer questions about one uploaded source. /no_think
Use only the document section below.
Do not include hidden reasoning, chain-of-thought, or <think> text.

Question: {question}

Document section:
{context}

Write concise notes that are directly useful for answering the question.
If this section has no relevant information, say: No relevant information in this section.

Section notes:"""
)


EXHAUSTIVE_FINAL_PROMPT = PromptTemplate.from_template(
    """You answer questions about one uploaded source. /no_think
Use only the notes below. These notes were produced by reading the source in sections, so they may cover different parts of the document.
Do not include hidden reasoning, chain-of-thought, or <think> text.
Keep the answer practical and mention when different sections add different details.
If the notes do not contain enough information, say that the source does not contain enough information.

Question: {question}

Notes from the full document:
{context}

Answer:"""
)


QUIZ_BATCH_PROMPT = PromptTemplate.from_template(
    """You are preparing a document-grounded quiz. /no_think
Use only the document section below.
Do not include hidden reasoning, chain-of-thought, or <think> text.

Quiz type: {quiz_type}
Difficulty: {difficulty}
Target final question count: {question_count}

Document section:
{context}

Extract useful quiz material from this section. Include important facts, concepts, definitions, procedures, examples, numbers, names, dates, warnings, or relationships.
Prefer material that can support clear questions. If this section has no useful quiz material, say: No useful quiz material.

Quiz material:"""
)


QUIZ_FINAL_PROMPT = PromptTemplate.from_template(
    """Create a quiz from one uploaded source. /no_think
Use only the document material below.
Do not include hidden reasoning, chain-of-thought, or <think> text.

Quiz type: {quiz_type}
Difficulty: {difficulty}
Number of questions: {question_count}

Rules:
- Generate exactly {question_count} questions.
- Keep every question answerable from the document material.
- Spread questions across the available material when possible.
- Do not mention information that is not supported by the material.
- For QCM/MCQ questions, provide exactly 4 options labeled A-D, the correct option letter, and a short explanation.
- For context/open questions, provide the expected answer and a short explanation.
- If difficulty is Mixed, include a balanced mix of easy, medium, and hard questions.
- If quiz type is Mixed, include both QCM/MCQ and context/open questions.

Format:
Return JSON only. Do not wrap it in markdown fences.
Use this schema:
[
  {{
    "type": "qcm",
    "difficulty": "Medium",
    "question": "Question text",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "explanation": "Short explanation"
  }},
  {{
    "type": "open",
    "difficulty": "Medium",
    "question": "Question text",
    "expected_answer": "Expected answer",
    "explanation": "Short explanation"
  }}
]

Use "qcm" for QCM/MCQ questions and "open" for context/open questions.
For QCM / multiple choice quiz type, every item must be type "qcm".
For Context questions quiz type, every item must be type "open".

Document material:
{context}

JSON:"""
)


def select_summary_docs(docs: list[Document], summary_chunks: int) -> list[Document]:
    if summary_chunks <= 0:
        return docs

    if len(docs) <= summary_chunks:
        return docs

    selected_indexes = {
        round(index * (len(docs) - 1) / (summary_chunks - 1))
        for index in range(summary_chunks)
    }
    return [docs[index] for index in sorted(selected_indexes)]


def build_summary_prompt(docs: Iterable[Document], summary_chunks: int, summary_max_chars: int) -> str:
    selected_docs = select_summary_docs(list(docs), summary_chunks)
    context = "\n\n".join(doc.page_content for doc in selected_docs)
    if summary_max_chars > 0:
        context = context[:summary_max_chars]
    source_type = "source"
    if selected_docs:
        source_type = selected_docs[0].metadata.get("source_type", "source")
        if source_type == "youtube":
            source_type = "YouTube video transcript"
        elif source_type == "pdf":
            source_type = "PDF"
    return SUMMARY_PROMPT.format(source_type=source_type, context=context)


def stream_llm_text(llm, prompt: str):
    for chunk in llm.stream(prompt):
        content = getattr(chunk, "content", chunk)
        if content:
            yield str(content)


def get_llm_text(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    return str(content or "")


def should_search_full_document(question: str) -> bool:
    lowered = question.lower()
    broad_phrases = (
        "all contents",
        "all content",
        "entire document",
        "entire pdf",
        "whole document",
        "whole pdf",
        "full document",
        "full pdf",
        "everything",
        "the rest",
        "rest of",
        "complete summary",
        "summarize all",
        "summarise all",
    )
    return any(phrase in lowered for phrase in broad_phrases)


def doc_label(doc: Document) -> str:
    chunk = doc.metadata.get("chunk", "?")
    return f"[Chunk {chunk}]\n{doc.page_content}"


def pack_docs_for_context(docs: Iterable[Document], max_context_chars: int) -> list[str]:
    batches = []
    current_parts = []
    current_length = 0

    for doc in docs:
        text = doc_label(doc)
        separator_length = 2 if current_parts else 0
        if current_parts and current_length + separator_length + len(text) > max_context_chars:
            batches.append("\n\n".join(current_parts))
            current_parts = []
            current_length = 0

        if len(text) > max_context_chars:
            text = text[:max_context_chars]

        current_parts.append(text)
        current_length += (2 if current_length else 0) + len(text)

    if current_parts:
        batches.append("\n\n".join(current_parts))

    return batches


def retrieve_context(vector_store, question: str, retrieval_k: int, max_context_chars: int) -> tuple[str, list[Document]]:
    start = time.perf_counter()
    docs = vector_store.similarity_search(question, k=retrieval_k)
    elapsed = time.perf_counter() - start
    logger.info("Retrieved %s chunks in %.2fs", len(docs), elapsed)

    context_parts = []
    remaining_chars = max_context_chars
    for doc in docs:
        if remaining_chars <= 0:
            break
        content = doc.page_content[:remaining_chars]
        context_parts.append(content)
        remaining_chars -= len(content)

    return "\n\n".join(context_parts), docs


def answer_from_full_document(llm, question: str, docs: list[Document], max_context_chars: int) -> tuple[str, list[str]]:
    batches = pack_docs_for_context(docs, max_context_chars)
    section_notes = []

    for batch_number, context in enumerate(batches, start=1):
        prompt = EXHAUSTIVE_BATCH_PROMPT.format(question=question, context=context)
        notes = get_llm_text(llm, prompt).strip()
        if notes and "no relevant information in this section" not in notes.lower():
            section_notes.append(f"[Section {batch_number}]\n{notes}")

    if not section_notes:
        section_notes = ["No section contained relevant information for the question."]

    while len("\n\n".join(section_notes)) > max_context_chars and len(section_notes) > 1:
        reduced_notes = []
        for context in pack_docs_for_context(
            [Document(page_content=notes, metadata={"chunk": index}) for index, notes in enumerate(section_notes)],
            max_context_chars,
        ):
            prompt = EXHAUSTIVE_BATCH_PROMPT.format(question=question, context=context)
            reduced_notes.append(get_llm_text(llm, prompt).strip())
        section_notes = [notes for notes in reduced_notes if notes]

    final_context = "\n\n".join(section_notes)[:max_context_chars]
    final_prompt = EXHAUSTIVE_FINAL_PROMPT.format(question=question, context=final_context)
    return get_llm_text(llm, final_prompt), batches


def build_quiz_context(docs: list[Document], question_count: int, max_context_chars: int) -> str:
    if not docs:
        return ""

    selected_count = min(len(docs), max(12, question_count * 4))
    selected_docs = select_summary_docs(docs, selected_count)
    context_parts = []
    remaining_chars = max_context_chars

    for doc in selected_docs:
        if remaining_chars <= 0:
            break
        text = doc_label(doc)
        if len(text) > remaining_chars:
            text = text[:remaining_chars]
        context_parts.append(text)
        remaining_chars -= len(text) + 2

    return "\n\n".join(context_parts)


def generate_quiz(
    llm,
    docs: list[Document],
    question_count: int,
    quiz_type: str,
    difficulty: str,
    max_context_chars: int,
) -> tuple[str, list[str]]:
    final_context = build_quiz_context(docs, question_count, max_context_chars)
    final_prompt = QUIZ_FINAL_PROMPT.format(
        quiz_type=quiz_type,
        difficulty=difficulty,
        question_count=question_count,
        context=final_context,
    )
    return get_llm_text(llm, final_prompt), [final_context] if final_context else []


def clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def parse_quiz_items(text: str) -> list[dict]:
    try:
        raw_items = json.loads(clean_json_text(text))
    except json.JSONDecodeError:
        return []

    if not isinstance(raw_items, list):
        return []

    quiz_items = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).strip().lower()
        question = str(item.get("question", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        difficulty = str(item.get("difficulty", "")).strip() or "Mixed"
        if not question:
            continue

        if item_type in {"qcm", "mcq", "multiple_choice", "multiple choice"}:
            options = item.get("options", {})
            if not isinstance(options, dict):
                continue
            normalized_options = {
                letter: str(options.get(letter, "")).strip()
                for letter in ("A", "B", "C", "D")
            }
            answer = str(item.get("answer", "")).strip().upper()[:1]
            if answer not in normalized_options or not all(normalized_options.values()):
                continue
            quiz_items.append(
                {
                    "type": "qcm",
                    "difficulty": difficulty,
                    "question": question,
                    "options": normalized_options,
                    "answer": answer,
                    "explanation": explanation,
                }
            )
        else:
            expected_answer = str(item.get("expected_answer") or item.get("answer") or "").strip()
            if not expected_answer:
                continue
            quiz_items.append(
                {
                    "type": "open",
                    "difficulty": difficulty,
                    "question": question,
                    "expected_answer": expected_answer,
                    "explanation": explanation,
                }
            )

    return quiz_items


def render_interactive_quiz(quiz_items: list[dict], quiz_key: str) -> None:
    if not quiz_items:
        return

    score = 0
    answered_qcm_count = 0

    for index, item in enumerate(quiz_items, start=1):
        item_key = f"{quiz_key}_{index}"
        item_type = item.get("type")
        difficulty = item.get("difficulty", "Mixed")

        st.markdown(f"**{index}. {difficulty}**")
        st.write(item["question"])

        if item_type == "qcm":
            options = item["options"]
            labels = [f"{letter}. {text}" for letter, text in options.items()]
            selected_label = st.radio(
                "Choose your answer",
                labels,
                key=f"quiz_choice_{item_key}",
                index=None,
            )
            submitted = st.button("Check answer", key=f"quiz_submit_{item_key}")
            if submitted:
                st.session_state[f"quiz_checked_{item_key}"] = True

            if st.session_state.get(f"quiz_checked_{item_key}"):
                if not selected_label:
                    st.warning("Choose one option first.")
                else:
                    selected_letter = selected_label.split(".", 1)[0]
                    correct_letter = item["answer"]
                    correct_text = options[correct_letter]
                    answered_qcm_count += 1
                    if selected_letter == correct_letter:
                        score += 1
                        st.success("Correct.")
                    else:
                        st.error(f"Not quite. The correct answer is {correct_letter}. {correct_text}")
                    if item.get("explanation"):
                        st.info(item["explanation"])
        else:
            st.text_area("Your answer", key=f"quiz_open_answer_{item_key}", height=90)
            if st.button("Show expected answer", key=f"quiz_reveal_{item_key}"):
                st.session_state[f"quiz_revealed_{item_key}"] = True
            if st.session_state.get(f"quiz_revealed_{item_key}"):
                st.success(f"Expected answer: {item['expected_answer']}")
                if item.get("explanation"):
                    st.info(item["explanation"])

        st.divider()

    if answered_qcm_count:
        st.caption(f"Checked QCM score: {score}/{answered_qcm_count}")


def build_qa_prompt(question: str, context: str) -> str:
    return QA_PROMPT.format(context=context, question=question)


def get_app_config():
    provider = get_config_value("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip()
    embedding_backend = get_config_value("EMBEDDING_BACKEND", DEFAULT_EMBEDDING_BACKEND).strip().lower()
    default_embedding_model = DEFAULT_OPENAI_EMBEDDING_MODEL if embedding_backend == "openai" else DEFAULT_EMBEDDING_MODEL
    embedding_model = get_config_value(
        "EMBEDDING_MODEL",
        get_config_value("OPENAI_EMBEDDING_MODEL", default_embedding_model),
    )
    normalized_embedding_model = EMBEDDING_MODEL_ALIASES.get(embedding_model.strip(), embedding_model.strip())
    temperature = float(get_config_value("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    max_output_tokens = int(get_config_value("MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)))
    retrieval_k = int(get_config_value("RETRIEVAL_K", str(DEFAULT_RETRIEVAL_K)))
    max_context_chars = int(get_config_value("MAX_CONTEXT_CHARS", str(DEFAULT_MAX_CONTEXT_CHARS)))
    chunk_size = int(get_config_value("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    chunk_overlap = int(get_config_value("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))
    summary_chunks = int(get_config_value("SUMMARY_CHUNKS", str(DEFAULT_SUMMARY_CHUNKS)))
    summary_max_chars = int(get_config_value("SUMMARY_MAX_CHARS", str(DEFAULT_SUMMARY_MAX_CHARS)))
    exhaustive_batch_chars = int(get_config_value("EXHAUSTIVE_BATCH_CHARS", str(DEFAULT_EXHAUSTIVE_BATCH_CHARS)))
    youtube_languages = get_config_value("YOUTUBE_TRANSCRIPT_LANGUAGES", DEFAULT_YOUTUBE_LANGUAGES)
    youtube_whisper_model = get_config_value("YOUTUBE_WHISPER_MODEL", DEFAULT_YOUTUBE_WHISPER_MODEL)
    youtube_max_whisper_minutes = int(
        get_config_value("YOUTUBE_MAX_WHISPER_MINUTES", str(DEFAULT_YOUTUBE_MAX_WHISPER_MINUTES))
    )

    provider_key = provider.lower()
    if provider_key == "openai":
        provider = "OpenAI"
        model_name = get_config_value("OPENAI_MODEL", "gpt-4o-mini")
        api_key = get_secret("OPENAI_API_KEY")
    elif provider_key in {"gemini", "google", "google ai", "google generative ai"}:
        provider = "Gemini"
        model_name = get_config_value("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    elif provider_key in {"hugging face", "huggingface"}:
        provider = "Hugging Face"
        model_name = get_config_value("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        api_key = get_secret("HUGGINGFACEHUB_API_TOKEN")
    else:
        provider = "Ollama"
        model_name = get_config_value("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        api_key = None

    return (
        provider,
        model_name,
        api_key,
        embedding_backend,
        normalized_embedding_model,
        temperature,
        max_output_tokens,
        retrieval_k,
        max_context_chars,
        chunk_size,
        chunk_overlap,
        summary_chunks,
        summary_max_chars,
        exhaustive_batch_chars,
        youtube_languages,
        youtube_whisper_model,
        youtube_max_whisper_minutes,
    )


def ensure_session_defaults() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("summary", "")
    st.session_state.setdefault("summary_key", "")
    st.session_state.setdefault("quiz", "")
    st.session_state.setdefault("quiz_items", [])
    st.session_state.setdefault("quiz_key", "")
    st.session_state.setdefault("quiz_batch_count", 0)
    st.session_state.setdefault("active_file_hash", "")
    st.session_state.setdefault("active_index_key", "")
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("youtube_url", "")


def render_quiz_controls(max_output_tokens: int) -> tuple[str, str, int]:
    control_cols = st.columns([1, 1, 1])
    with control_cols[0]:
        quiz_type = st.selectbox(
            "Question type",
            ["QCM / multiple choice", "Context questions", "Mixed"],
            help="QCM creates four-option multiple-choice questions. Context questions are open-ended.",
        )
    with control_cols[1]:
        difficulty = st.selectbox(
            "Difficulty",
            ["Easy", "Medium", "Hard", "Mixed"],
            index=1,
        )
    with control_cols[2]:
        question_count = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=30,
            value=DEFAULT_QUIZ_QUESTION_COUNT,
            step=1,
        )

    if question_count > 8 and max_output_tokens < 1200:
        st.warning(
            "Your current MAX_OUTPUT_TOKENS is low for a large quiz. The model may stop early; increase it in .env for longer quizzes.",
            icon="!",
        )

    return quiz_type, difficulty, int(question_count)


def main() -> None:
    init_page()
    ensure_session_defaults()

    (
        provider,
        model_name,
        api_key,
        embedding_backend,
        embedding_model,
        temperature,
        max_output_tokens,
        retrieval_k,
        max_context_chars,
        chunk_size,
        chunk_overlap,
        summary_chunks,
        summary_max_chars,
        exhaustive_batch_chars,
        youtube_languages,
        youtube_whisper_model,
        youtube_max_whisper_minutes,
    ) = get_app_config()
    source_type = st.radio("Source", ["PDF", "YouTube"], horizontal=True)

    payload = None
    if source_type == "PDF":
        uploaded_file = st.file_uploader(
            "Drop your PDF here",
            type=["pdf"],
            accept_multiple_files=False,
            help="Drag and drop a PDF file here, or click Browse files.",
            key=f"pdf_uploader_{st.session_state.uploader_key}",
        )

        if not uploaded_file:
            st.info("Drop a PDF file above to generate a summary and start chatting with it.")
            return

        with st.spinner("Reading PDF..."):
            payload = read_pdf(uploaded_file)

        if not payload.text:
            st.error("I could not extract text from this PDF. It may be scanned or image-only.")
            return
    else:
        youtube_url = st.text_input(
            "YouTube video link",
            placeholder="https://www.youtube.com/watch?v=...",
            key="youtube_url",
        )
        st.caption("kom dak video thom pek ors luy nh.")
        max_whisper_minutes = youtube_max_whisper_minutes
        transcript_languages = youtube_languages

        if not youtube_url:
            st.info("Paste a YouTube video link above to fetch captions or transcribe audio.")
            return

        languages = [language.strip() for language in transcript_languages.split(",") if language.strip()]
        try:
            with st.spinner("Reading YouTube transcript..."):
                payload = read_youtube(
                    youtube_url,
                    languages or [DEFAULT_YOUTUBE_LANGUAGES],
                    True,
                    youtube_whisper_model,
                    int(max_whisper_minutes),
                )
        except Exception as exc:
            logger.exception("YouTube ingestion failed")
            st.error(f"YouTube ingestion failed: {exc}")
            return

    if payload.file_hash != st.session_state.active_file_hash:
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.summary_key = ""
        st.session_state.quiz = ""
        st.session_state.quiz_items = []
        st.session_state.quiz_key = ""
        st.session_state.quiz_batch_count = 0
        st.session_state.active_file_hash = payload.file_hash

    docs = split_source(payload, chunk_size, chunk_overlap)

    col_meta, col_summary = st.columns([1, 2])
    with col_meta:
        st.subheader("Source")
        st.write(f"**Name:** {payload.source_name}")
        st.write(f"**Type:** {payload.source_type.title()}")
        if payload.url:
            st.write(f"**URL:** {payload.url}")
        if payload.duration_seconds:
            st.write(f"**Duration:** {format_timestamp(payload.duration_seconds)}")
        if payload.transcript_source:
            st.write(f"**Transcript:** {payload.transcript_source}")
        st.write(f"**Items:** {payload.item_count}")
        st.write(f"**Chunks:** {len(docs)}")

    try:
        index_key = f"{payload.source_type}:{payload.file_hash}:{embedding_backend}:{embedding_model}:{chunk_size}:{chunk_overlap}"
        with st.spinner("Indexing source..."):
            if index_key != st.session_state.active_index_key:
                logger.info(
                    "Indexing %s items from %s into %s chunks",
                    payload.item_count,
                    payload.source_name,
                    len(docs),
                )
                st.session_state.vector_store = build_vector_store(
                    payload.source_type,
                    payload.file_hash,
                    embedding_backend,
                    embedding_model,
                    docs,
                )
                st.session_state.active_index_key = index_key
            vector_store = st.session_state.vector_store
        llm = get_llm(provider, model_name, temperature, api_key, max_output_tokens)
    except Exception as exc:
        logger.exception("Setup failed")
        st.error(f"Setup failed: {exc}")
        return

    with col_summary:
        st.subheader("Summary")
        summary_key = (
            f"summary-v2:{payload.source_type}:{payload.file_hash}:{provider}:{model_name}:"
            f"{summary_chunks}:{summary_max_chars}:{chunk_size}:{chunk_overlap}"
        )
        if st.session_state.summary and st.session_state.summary_key == summary_key:
            st.markdown(st.session_state.summary)
        else:
            try:
                logger.info("Starting summary with provider=%s model=%s", provider, model_name)
                prompt = build_summary_prompt(docs, summary_chunks, summary_max_chars)
                summary_box = st.empty()
                summary_parts = []
                with st.spinner("Summarizing source..."):
                    for token in stream_llm_text(llm, prompt):
                        summary_parts.append(token)
                        summary_box.markdown("".join(summary_parts))
                st.session_state.summary = "".join(summary_parts)
                st.session_state.summary_key = summary_key
                logger.info("Summary finished with %s characters", len(st.session_state.summary))
            except Exception as exc:
                logger.exception("Summary failed")
                st.error(f"Summary failed: {exc}")

    st.divider()
    chat_tab, quiz_tab = st.tabs(["Chat", "Quiz"])

    with quiz_tab:
        st.subheader("Create a quiz")
        quiz_type, difficulty, question_count = render_quiz_controls(max_output_tokens)

        if st.button("Generate quiz", type="primary", use_container_width=True):
            try:
                start = time.perf_counter()
                with st.spinner("Creating quiz from the whole source..."):
                    quiz, batches = generate_quiz(
                        llm,
                        docs,
                        question_count,
                        quiz_type,
                        difficulty,
                        max_context_chars,
                    )
                quiz_items = parse_quiz_items(quiz)
                quiz_key = hashlib.sha256(
                    f"{payload.file_hash}:{quiz_type}:{difficulty}:{question_count}:{quiz}".encode("utf-8")
                ).hexdigest()[:12]
                st.session_state.quiz = quiz
                st.session_state.quiz_items = quiz_items
                st.session_state.quiz_key = quiz_key
                st.session_state.quiz_batch_count = len(batches)
                logger.info(
                    "Quiz generated in %.2fs with %s batches and %s characters",
                    time.perf_counter() - start,
                    len(batches),
                    len(quiz),
                )
            except Exception as exc:
                logger.exception("Quiz generation failed")
                error_text = str(exc)
                if "429" in error_text or "quota" in error_text.lower() or "rate" in error_text.lower():
                    st.error(
                        "Quiz generation hit the Gemini free-tier quota. Wait for the retry time in the error, "
                        "try again later, or switch to another Gemini API key/billing plan."
                    )
                    with st.expander("Full quota error"):
                        st.write(error_text)
                else:
                    st.error(f"Quiz generation failed: {exc}")

        if st.session_state.quiz:
            if st.session_state.quiz_batch_count:
                st.caption(
                    f"Quiz generated from a sampled source context across {len(docs)} chunks "
                    f"using {st.session_state.quiz_batch_count} model request"
                    f"{'' if st.session_state.quiz_batch_count == 1 else 's'}."
                )
            if st.session_state.quiz_items:
                render_interactive_quiz(st.session_state.quiz_items, st.session_state.quiz_key)
            else:
                st.warning("I could not turn this quiz into interactive questions, so I am showing the raw quiz.")
                st.markdown(st.session_state.quiz)

    with chat_tab:
        st.subheader("Chat with this source")
        search_full_document = st.toggle(
            "Search whole source",
            value=False,
            help="Use this for broad questions that need every chunk. Broad prompts are also detected automatically. It is slower, but it reads the source in batches instead of only retrieving the nearest chunks.",
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about the uploaded source")
        if not question:
            return

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                answer_box = st.empty()
                answer_parts = []
                generation_start = time.perf_counter()
                exhaustive_search = search_full_document or should_search_full_document(question)

                if exhaustive_search:
                    with st.spinner("Reading the whole source in batches..."):
                        answer, batches = answer_from_full_document(llm, question, docs, exhaustive_batch_chars)
                    answer_box.markdown(answer)
                    logger.info(
                        "Full-document answer finished in %.2fs with %s batches and %s characters",
                        time.perf_counter() - generation_start,
                        len(batches),
                        len(answer),
                    )
                    with st.expander("Source batches scanned"):
                        st.caption(f"Scanned {len(docs)} chunks in {len(batches)} context-window-sized batches.")
                        for batch_number, batch in enumerate(batches, start=1):
                            st.caption(f"Batch {batch_number}")
                            st.write(batch[:1200])
                else:
                    with st.spinner("Searching the source..."):
                        context, sources = retrieve_context(vector_store, question, retrieval_k, max_context_chars)

                    prompt = build_qa_prompt(question, context)
                    logger.info("Starting answer with provider=%s model=%s", provider, model_name)
                    for token in stream_llm_text(llm, prompt):
                        answer_parts.append(token)
                        answer_box.markdown("".join(answer_parts))
                    answer = "".join(answer_parts)
                    logger.info("Answer finished in %.2fs with %s characters", time.perf_counter() - generation_start, len(answer))

                    with st.expander("Retrieved context"):
                        for source in sources:
                            chunk = source.metadata.get("chunk", "?")
                            st.caption(f"Chunk {chunk}")
                            st.write(source.page_content[:800])

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_history.append((question, answer))
        except Exception as exc:
            logger.exception("Chat failed")
            st.error(f"Chat failed: {exc}")


if __name__ == "__main__":
    main()

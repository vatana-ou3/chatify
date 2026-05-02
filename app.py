import hashlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate


APP_TITLE = "Chatify"
DEFAULT_EMBEDDING_BACKEND = "fastembed"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OLLAMA_MODEL = "frob/qwen3.5-instruct:4b"  # Change this to your Ollama model name if different
DEFAULT_LLM_PROVIDER = "Ollama"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 1200
DEFAULT_RETRIEVAL_K = 3
DEFAULT_MAX_CONTEXT_CHARS = 5000
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SUMMARY_CHUNKS = 16
DEFAULT_SUMMARY_MAX_CHARS = 24000
DEFAULT_EXHAUSTIVE_BATCH_CHARS = 5000
DEFAULT_QUIZ_QUESTION_COUNT = 5
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
class PdfPayload:
    file_hash: str
    file_name: str
    text: str
    page_count: int


def get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        value = st.secrets.get(name)
    except Exception:
        return None

    return str(value) if value else None


def get_config_value(name: str, default: str | None = None) -> str | None:
    return get_secret(name) or default


def init_page() -> None:
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, page_icon="PDF", layout="wide")
    title_col, action_col = st.columns([1, 0.18])
    with title_col:
        st.title(APP_TITLE)
        st.caption("Upload a PDF, get a concise summary, then ask questions grounded in the document.")
    with action_col:
        if st.button("New chat", use_container_width=True):
            reset_chat()
            st.rerun()


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.summary = ""
    st.session_state.quiz = ""
    st.session_state.quiz_batch_count = 0
    st.session_state.active_file_hash = ""
    st.session_state.active_index_key = ""
    st.session_state.vector_store = None
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


def read_pdf(uploaded_file) -> PdfPayload:
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

    return PdfPayload(
        file_hash=file_hash,
        file_name=uploaded_file.name,
        text="\n".join(pages).strip(),
        page_count=len(reader.pages),
    )


def split_pdf(payload: PdfPayload, chunk_size: int, chunk_overlap: int) -> list[Document]:
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
                "source": payload.file_name,
                "file_hash": payload.file_hash,
                "chunk": index,
            },
        )
        for index, chunk in enumerate(chunks)
    ]
    logger.info(
        "Split %s pages into %s chunks in %.2fs using chunk_size=%s chunk_overlap=%s",
        payload.page_count,
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


def build_vector_store(file_hash: str, embedding_backend: str, embedding_model: str, docs: list[Document]):
    embeddings = get_embeddings(embedding_backend, embedding_model)
    embedding_hash = hashlib.sha256(f"{embedding_backend}:{embedding_model}".encode("utf-8")).hexdigest()[:8]
    collection_name = f"pdf_{file_hash}_{embedding_hash}"
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
    ids = [f"{file_hash}-{doc.metadata['chunk']}" for doc in docs]
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
    """You are explaining a PDF for a reader. /no_think

Write an explanatory guide using only the document content below.
Do not include hidden reasoning, chain-of-thought, or <think> text.
Do not only say what the document is about. Explain the actual content so a reader can understand it without reading the PDF.

Use this structure:

## Plain-language explanation
Explain the document's ideas step by step. Define important terms, explain why each idea matters, and connect related points.

## Key details from the document
List the important facts, examples, tools, commands, names, dates, numbers, warnings, or procedures that appear in the content.

## What the reader should understand
Explain the main takeaways and how the pieces fit together.

## Good follow-up questions
Suggest questions the reader can ask to understand unclear or advanced parts.

Document content:
{context}

Summary:"""
)


QA_PROMPT = PromptTemplate.from_template(
    """You answer questions about one uploaded PDF. /no_think
Use only the context below. If the answer is not in the PDF, say that the document does not contain enough information.
Do not include hidden reasoning, chain-of-thought, or <think> text.
Keep the answer concise and practical.

Context:
{context}

Question: {question}

Answer:"""
)


EXHAUSTIVE_BATCH_PROMPT = PromptTemplate.from_template(
    """You answer questions about one uploaded PDF. /no_think
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
    """You answer questions about one uploaded PDF. /no_think
Use only the notes below. These notes were produced by reading the PDF in sections, so they may cover different parts of the document.
Do not include hidden reasoning, chain-of-thought, or <think> text.
Keep the answer practical and mention when different sections add different details.
If the notes do not contain enough information, say that the document does not contain enough information.

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
    """Create a quiz from one uploaded PDF. /no_think
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
- For QCM/MCQ questions, provide 4 options labeled A-D, mark the correct answer, and add a short explanation.
- For context/open questions, provide the expected answer and a short explanation.
- If difficulty is Mixed, include a balanced mix of easy, medium, and hard questions.
- If quiz type is Mixed, include both QCM/MCQ and context/open questions.

Format:
## Quiz

### 1. [Question type] [Difficulty]
Question text

Options:
A. ...
B. ...
C. ...
D. ...

Answer: ...
Explanation: ...

For context/open questions, omit the Options block.

Document material:
{context}

Quiz:"""
)


def select_summary_docs(docs: list[Document], summary_chunks: int) -> list[Document]:
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
    context = context[:summary_max_chars]
    return SUMMARY_PROMPT.format(context=context)


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


def generate_quiz(
    llm,
    docs: list[Document],
    question_count: int,
    quiz_type: str,
    difficulty: str,
    max_context_chars: int,
) -> tuple[str, list[str]]:
    batches = pack_docs_for_context(docs, max_context_chars)
    material_parts = []

    for batch_number, context in enumerate(batches, start=1):
        prompt = QUIZ_BATCH_PROMPT.format(
            quiz_type=quiz_type,
            difficulty=difficulty,
            question_count=question_count,
            context=context,
        )
        material = get_llm_text(llm, prompt).strip()
        if material and "no useful quiz material" not in material.lower():
            material_parts.append(f"[Section {batch_number}]\n{material}")

    if not material_parts:
        material_parts = ["No useful quiz material was found in the extracted document text."]

    while len("\n\n".join(material_parts)) > max_context_chars and len(material_parts) > 1:
        reduced_parts = []
        for context in pack_docs_for_context(
            [Document(page_content=material, metadata={"chunk": index}) for index, material in enumerate(material_parts)],
            max_context_chars,
        ):
            prompt = QUIZ_BATCH_PROMPT.format(
                quiz_type=quiz_type,
                difficulty=difficulty,
                question_count=question_count,
                context=context,
            )
            reduced = get_llm_text(llm, prompt).strip()
            if reduced:
                reduced_parts.append(reduced)
        material_parts = reduced_parts or material_parts[:1]

    final_context = "\n\n".join(material_parts)[:max_context_chars]
    final_prompt = QUIZ_FINAL_PROMPT.format(
        quiz_type=quiz_type,
        difficulty=difficulty,
        question_count=question_count,
        context=final_context,
    )
    return get_llm_text(llm, final_prompt), batches


def build_qa_prompt(question: str, context: str) -> str:
    return QA_PROMPT.format(context=context, question=question)


def get_app_config():
    provider = get_config_value("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
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

    if provider == "OpenAI":
        model_name = get_config_value("OPENAI_MODEL", "gpt-4o-mini")
        api_key = get_secret("OPENAI_API_KEY")
    elif provider == "Hugging Face":
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
    )


def ensure_session_defaults() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("summary", "")
    st.session_state.setdefault("quiz", "")
    st.session_state.setdefault("quiz_batch_count", 0)
    st.session_state.setdefault("active_file_hash", "")
    st.session_state.setdefault("active_index_key", "")
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("uploader_key", 0)


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
    ) = get_app_config()
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

    if payload.file_hash != st.session_state.active_file_hash:
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.summary = ""
        st.session_state.quiz = ""
        st.session_state.quiz_batch_count = 0
        st.session_state.active_file_hash = payload.file_hash

    docs = split_pdf(payload, chunk_size, chunk_overlap)

    col_meta, col_summary = st.columns([1, 2])
    with col_meta:
        st.subheader("Document")
        st.write(f"**File:** {payload.file_name}")
        st.write(f"**Pages:** {payload.page_count}")
        st.write(f"**Chunks:** {len(docs)}")

    try:
        index_key = f"{payload.file_hash}:{embedding_backend}:{embedding_model}:{chunk_size}:{chunk_overlap}"
        with st.spinner("Indexing document..."):
            if index_key != st.session_state.active_index_key:
                logger.info(
                    "Indexing %s pages from %s into %s chunks",
                    payload.page_count,
                    payload.file_name,
                    len(docs),
                )
                st.session_state.vector_store = build_vector_store(payload.file_hash, embedding_backend, embedding_model, docs)
                st.session_state.active_index_key = index_key
            vector_store = st.session_state.vector_store
        llm = get_llm(provider, model_name, temperature, api_key, max_output_tokens)
    except Exception as exc:
        logger.exception("Setup failed")
        st.error(f"Setup failed: {exc}")
        return

    with col_summary:
        st.subheader("Summary")
        if not st.session_state.summary:
            try:
                logger.info("Starting summary with provider=%s model=%s", provider, model_name)
                prompt = build_summary_prompt(docs, summary_chunks, summary_max_chars)
                summary_box = st.empty()
                summary_parts = []
                with st.spinner("Summarizing PDF..."):
                    for token in stream_llm_text(llm, prompt):
                        summary_parts.append(token)
                        summary_box.markdown("".join(summary_parts))
                st.session_state.summary = "".join(summary_parts)
                logger.info("Summary finished with %s characters", len(st.session_state.summary))
            except Exception as exc:
                logger.exception("Summary failed")
                st.error(f"Summary failed: {exc}")
        else:
            st.markdown(st.session_state.summary)

    st.divider()
    chat_tab, quiz_tab = st.tabs(["Chat", "Quiz"])

    with quiz_tab:
        st.subheader("Create a quiz")
        quiz_type, difficulty, question_count = render_quiz_controls(max_output_tokens)

        if st.button("Generate quiz", type="primary", use_container_width=True):
            try:
                start = time.perf_counter()
                with st.spinner("Creating quiz from the whole document..."):
                    quiz, batches = generate_quiz(
                        llm,
                        docs,
                        question_count,
                        quiz_type,
                        difficulty,
                        exhaustive_batch_chars,
                    )
                st.session_state.quiz = quiz
                st.session_state.quiz_batch_count = len(batches)
                logger.info(
                    "Quiz generated in %.2fs with %s batches and %s characters",
                    time.perf_counter() - start,
                    len(batches),
                    len(quiz),
                )
            except Exception as exc:
                logger.exception("Quiz generation failed")
                st.error(f"Quiz generation failed: {exc}")

        if st.session_state.quiz:
            if st.session_state.quiz_batch_count:
                st.caption(f"Quiz generated from {len(docs)} chunks across {st.session_state.quiz_batch_count} batches.")
            st.markdown(st.session_state.quiz)

    with chat_tab:
        st.subheader("Chat with this PDF")
        search_full_document = st.toggle(
            "Search whole document",
            value=False,
            help="Use this for broad questions that need every chunk. Broad prompts are also detected automatically. It is slower, but it reads the PDF in batches instead of only retrieving the nearest chunks.",
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about the uploaded PDF")
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
                    with st.spinner("Reading the whole PDF in batches..."):
                        answer, batches = answer_from_full_document(llm, question, docs, exhaustive_batch_chars)
                    answer_box.markdown(answer)
                    logger.info(
                        "Full-document answer finished in %.2fs with %s batches and %s characters",
                        time.perf_counter() - generation_start,
                        len(batches),
                        len(answer),
                    )
                    with st.expander("Document batches scanned"):
                        st.caption(f"Scanned {len(docs)} chunks in {len(batches)} context-window-sized batches.")
                        for batch_number, batch in enumerate(batches, start=1):
                            st.caption(f"Batch {batch_number}")
                            st.write(batch[:1200])
                else:
                    with st.spinner("Searching the PDF..."):
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

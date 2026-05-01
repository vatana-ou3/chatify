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
DEFAULT_OLLAMA_MODEL = "frob/qwen3.5-instruct:4b"  # Change this to your Ollama model name if different
DEFAULT_LLM_PROVIDER = "Ollama"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_RETRIEVAL_K = 3
DEFAULT_MAX_CONTEXT_CHARS = 5000
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SUMMARY_CHUNKS = 16
DEFAULT_SUMMARY_MAX_CHARS = 24000
CHROMA_DIR = ".chroma"
FASTEMBED_CACHE_DIR = ".fastembed_cache"
EMBEDDING_MODEL_ALIASES = {
    "bge-m3": "BAAI/bge-m3",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/bge-m3": "BAAI/bge-m3",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfPayload:
    file_hash: str
    file_name: str
    text: str
    page_count: int


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
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
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
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    if provider == "Hugging Face":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            temperature=temperature,
            max_new_tokens=max_output_tokens,
            huggingfacehub_api_token=api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
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


def build_qa_prompt(question: str, context: str) -> str:
    return QA_PROMPT.format(context=context, question=question)


def get_app_config():
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    embedding_backend = os.getenv("EMBEDDING_BACKEND", DEFAULT_EMBEDDING_BACKEND).strip().lower()
    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    normalized_embedding_model = EMBEDDING_MODEL_ALIASES.get(embedding_model.strip(), embedding_model.strip())
    temperature = float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE))
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))
    retrieval_k = int(os.getenv("RETRIEVAL_K", DEFAULT_RETRIEVAL_K))
    max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS))
    chunk_size = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    summary_chunks = int(os.getenv("SUMMARY_CHUNKS", DEFAULT_SUMMARY_CHUNKS))
    summary_max_chars = int(os.getenv("SUMMARY_MAX_CHARS", DEFAULT_SUMMARY_MAX_CHARS))

    if provider == "OpenAI":
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "Hugging Face":
        model_name = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    else:
        provider = "Ollama"
        model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
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
    )


def ensure_session_defaults() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("summary", "")
    st.session_state.setdefault("active_file_hash", "")
    st.session_state.setdefault("active_index_key", "")
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("uploader_key", 0)


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
    st.subheader("Chat with this PDF")

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
            with st.spinner("Searching the PDF..."):
                context, sources = retrieve_context(vector_store, question, retrieval_k, max_context_chars)

            answer_box = st.empty()
            answer_parts = []
            prompt = build_qa_prompt(question, context)
            logger.info("Starting answer with provider=%s model=%s", provider, model_name)
            generation_start = time.perf_counter()
            for token in stream_llm_text(llm, prompt):
                answer_parts.append(token)
                answer_box.markdown("".join(answer_parts))
            answer = "".join(answer_parts)
            logger.info("Answer finished in %.2fs with %s characters", time.perf_counter() - generation_start, len(answer))

            if sources:
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

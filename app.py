import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate


APP_TITLE = "Chatify"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_OLLAMA_MODEL = "frob/qwen3.5-instruct:4b"  # Change this to your Ollama model name if different
DEFAULT_LLM_PROVIDER = "Ollama"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 512
CHROMA_DIR = ".chroma"
EMBEDDING_MODEL_ALIASES = {
    "bge-m3": "BAAI/bge-m3",
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


def split_pdf(payload: PdfPayload) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(payload.text)
    return [
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


@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str):
    normalized_model_name = EMBEDDING_MODEL_ALIASES.get(model_name.strip(), model_name.strip())
    model_kwargs = {"device": "cpu"}
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if hf_token:
        model_kwargs["token"] = hf_token

    return HuggingFaceEmbeddings(
        model_name=normalized_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(file_hash: str, embedding_model: str, docs: list[Document]):
    embeddings = get_embeddings(embedding_model)
    embedding_hash = hashlib.sha256(embedding_model.encode("utf-8")).hexdigest()[:8]
    collection_name = f"pdf_{file_hash}_{embedding_hash}"
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    if vector_store._collection.count() > 0:
        logger.info("Loaded existing Chroma collection %s", collection_name)
        return vector_store

    logger.info(
        "Creating Chroma collection %s with %s chunks using %s",
        collection_name,
        len(docs),
        embedding_model,
    )
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )


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
    """You are summarizing a PDF for a reader. /no_think

Write a concise summary using only the document content below.
Do not include hidden reasoning, chain-of-thought, or <think> text.
Keep the whole answer under 180 words.
Include:
- The main topic
- 3 to 5 important points
- Any key names, dates, numbers, tools, or actions if present

Document content:
{context}

Summary:"""
)


QA_PROMPT = PromptTemplate.from_template(
    """You answer questions about one uploaded PDF.
Use only the context below. If the answer is not in the PDF, say that the document does not contain enough information.

Context:
{context}

Question: {question}

Answer:"""
)


def build_summary_prompt(docs: Iterable[Document]) -> str:
    selected_docs = list(docs)[:8]
    context = "\n\n".join(doc.page_content for doc in selected_docs)
    context = context[:8000]
    return SUMMARY_PROMPT.format(context=context)


def stream_llm_text(llm, prompt: str):
    for chunk in llm.stream(prompt):
        content = getattr(chunk, "content", chunk)
        if content:
            yield str(content)


def make_chat_chain(llm, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )


def get_app_config():
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    normalized_embedding_model = EMBEDDING_MODEL_ALIASES.get(embedding_model.strip(), embedding_model.strip())
    temperature = float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE))
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))

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

    return provider, model_name, api_key, normalized_embedding_model, temperature, max_output_tokens


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

    provider, model_name, api_key, embedding_model, temperature, max_output_tokens = get_app_config()
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key=f"pdf_uploader_{st.session_state.uploader_key}")

    if not uploaded_file:
        st.info("Choose a PDF to begin.")
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

    docs = split_pdf(payload)

    col_meta, col_summary = st.columns([1, 2])
    with col_meta:
        st.subheader("Document")
        st.write(f"**File:** {payload.file_name}")
        st.write(f"**Pages:** {payload.page_count}")
        st.write(f"**Chunks:** {len(docs)}")

    try:
        index_key = f"{payload.file_hash}:{embedding_model}"
        with st.spinner("Indexing document..."):
            if index_key != st.session_state.active_index_key:
                logger.info(
                    "Indexing %s pages from %s into %s chunks",
                    payload.page_count,
                    payload.file_name,
                    len(docs),
                )
                st.session_state.vector_store = build_vector_store(payload.file_hash, embedding_model, docs)
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
                prompt = build_summary_prompt(docs)
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
        chat_chain = make_chat_chain(llm, vector_store)
        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF..."):
                response = chat_chain.invoke(
                    {
                        "question": question,
                        "chat_history": st.session_state.chat_history,
                    }
                )
            answer = response["answer"]
            st.markdown(answer)

            sources = response.get("source_documents", [])
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

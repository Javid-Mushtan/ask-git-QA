import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from utils.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, VECTORSTORE_DIR,
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
)

EXTENSION_TO_LANGUAGE = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".jsx":   Language.JS,
    ".ts":    Language.JS,
    ".tsx":   Language.JS,
    ".java":  Language.JAVA,
    ".go":    Language.GO,
    ".rb":    Language.RUBY,
    ".rs":    Language.RUST,
    ".cpp":   Language.CPP,
    ".cc":    Language.CPP,
    ".c":     Language.C,
    ".cs":    Language.CSHARP,
    ".html":  Language.HTML,
    ".md":    Language.MARKDOWN,
    ".sol":   Language.SOL,
}


def get_splitter_for_file(file_path: str) -> RecursiveCharacterTextSplitter:
    ext = os.path.splitext(file_path.lower())[1]
    language = EXTENSION_TO_LANGUAGE.get(ext)

    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def split_documents(documents: list) -> list:
    chunks = []
    for doc in documents:
        source = doc.metadata.get("source", "")
        ext = os.path.splitext(source.lower())[1]
        language = EXTENSION_TO_LANGUAGE.get(ext)

        splitter = get_splitter_for_file(source)
        doc_chunks = splitter.split_documents([doc])

        for chunk in doc_chunks:
            chunk.metadata["language"] = language.value if language else "text"

        chunks.extend(doc_chunks)

    print(f"[chunker] {len(documents)} docs → {len(chunks)} chunks")
    return chunks


def get_embeddings():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key,base_url="https://openrouter.ai/api/v1")


def create_vector_database(documents: list, repo_id: str) -> Chroma:
    chunks = split_documents(documents)

    embeddings = get_embeddings()
    persist_dir = os.path.join(VECTORSTORE_DIR, repo_id)

    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
            print("Existing vector store removed.")
        except Exception as e:
            print(f"Warning: could not remove directory: {e}")

    total = len(chunks)
    first_batch = chunks[:EMBEDDING_BATCH_SIZE]
    db = Chroma.from_documents(first_batch, embeddings, persist_directory=persist_dir)

    for i in range(EMBEDDING_BATCH_SIZE, total, EMBEDDING_BATCH_SIZE):
        batch = chunks[i: i + EMBEDDING_BATCH_SIZE]
        print(f"Embedding batch {i}–{min(i + EMBEDDING_BATCH_SIZE, total)}…")
        db.add_documents(batch)

    print(f"Vector store ready for {repo_id}  ({total} chunks)")
    return db


def load_vector_store(repo_id: str) -> Chroma:
    embeddings = get_embeddings()
    persist_dir = os.path.join(VECTORSTORE_DIR, repo_id)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
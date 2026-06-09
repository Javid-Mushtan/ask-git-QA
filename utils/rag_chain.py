from langchain.chains.combine_documents import create_stuff_documents_chain

from utils.config import TOP_K
from utils.llm import get_llm
from utils.prompt import prompt
from utils.vector_database import load_vector_store


def get_qa_chain(repo_id):
    db = load_vector_store(repo_id)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    qa_chain = create_stuff_documents_chain(get_llm(),prompt)

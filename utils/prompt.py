from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are an intelligent chatbot.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.

{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ]
)
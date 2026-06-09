import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils.config import LANGUAGE_MODEL, TEMPERATURE, MAX_NEW_TOKENS

load_dotenv()

def get_llm(model_name=LANGUAGE_MODEL):
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise Exception("OPENROUTER_API_KEY missing in .env")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS
    )
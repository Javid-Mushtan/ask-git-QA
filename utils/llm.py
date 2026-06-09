import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sympy.physics.units import temperature

from utils.config import LANGUAGE_MODEL, TEMPERATURE, MAX_NEW_TOKENS

load_dotenv()

def get_llm(model_name=LANGUAGE_MODEL):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OpenAI API key not found. Please set it ")

    return ChatOpenAI(
        api_key=api_key,
        model_name=model_name,
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        base_url="https://openrouter.ai/api/v1"
    )
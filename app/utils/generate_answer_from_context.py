from app.utils import build_prompt
from app.LLM import chat_with_gemini

def generate_answer_from_context(query: str, context: str) -> str:
    prompt = build_prompt(query, context)
    answer = chat_with_gemini(prompt)
    return answer
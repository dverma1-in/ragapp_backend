import json
import google.generativeai as genai
from app.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

_model = genai.GenerativeModel("gemini-2.5-flash")

async def chat_with_gemini(prompt: str) -> str:
    """Plain text generation."""
    response = await _model.generate_content_async(prompt)
    return response.text


async def chat_with_gemini_json(prompt: str) -> dict:
    """
    Generation with JSON output expected.
    The prompt must instruct the model to respond ONLY with valid JSON.
    Falls back to raw text wrapped in {"raw": ...} if parsing fails.
    """
    response = await _model.generate_content_async(prompt)
    text = response.text.strip()

    # Strip markdown code fences if the model added them
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}

"""
Query transformer.

Takes the raw user query + conversation history and produces:
  1. A rewritten query  – resolves pronouns / references using history
  2. Two paraphrases    – alternate phrasings for broader retrieval
  3. A HyDE document    – a hypothetical ideal answer whose embedding
                          often lands closer to relevant chunks than
                          the question embedding itself

All four variants are used by the hybrid retriever in parallel.
The LLM is asked to respond with strict JSON so we can parse it reliably.
"""

from app.LLM import chat_with_gemini_json


_TRANSFORM_PROMPT = """\
You are a search query optimizer for a RAG (retrieval-augmented generation) system.

Given the conversation history and the latest user query, produce EXACTLY this JSON object and nothing else:

{{
  "rewritten":    "<rewritten query resolving all pronouns/references using the conversation history>",
  "paraphrase_1": "<first alternative phrasing of the query>",
  "paraphrase_2": "<second alternative phrasing of the query>",
  "hyde":         "<a 2-3 sentence hypothetical answer that a document in the knowledge base MIGHT contain if it were perfectly relevant to the query>"
}}

Rules:
- If there is no conversation history, rewritten == original query.
- Paraphrases must use different vocabulary but preserve the intent.
- The HyDE text should read like a factual document excerpt, NOT like an answer to a question.
- Respond ONLY with the JSON object. No markdown, no explanation.

Conversation history (may be empty):
{history}

Latest user query:
{query}
"""


async def transform_query(query: str, history_text: str) -> dict:
    """
    Returns dict with keys: rewritten, paraphrase_1, paraphrase_2, hyde.
    Falls back gracefully if the LLM returns unexpected output.
    """
    prompt = _TRANSFORM_PROMPT.format(history=history_text or "(none)", query=query)
    result = await chat_with_gemini_json(prompt)

    # Ensure all expected keys exist even if the model misbehaves
    fallback = {
        "rewritten":    query,
        "paraphrase_1": query,
        "paraphrase_2": query,
        "hyde":         query,
    }
    for key in fallback:
        if key not in result or not isinstance(result[key], str):
            result[key] = fallback[key]

    return result


def expand_to_query_list(transformed: dict) -> list[str]:
    """Return all four query variants as a flat list."""
    return [
        transformed["rewritten"],
        transformed["paraphrase_1"],
        transformed["paraphrase_2"],
        transformed["hyde"],
    ]

"""
Answer service – orchestrates the full advanced RAG pipeline.

Per-request flow:
  1. Load conversation memory for this session
  2. Transform query  →  rewritten + 2 paraphrases + HyDE variant
  3. Agentic retrieval loop
       a. Hybrid search  (dense + BM25)  across all query variants
       b. RRF merge
       c. Cross-encoder re-rank  → top-N child chunks
       d. Parent-doc expansion   → full-context parent chunks
       e. Sufficiency check      → if not sufficient, reformulate & retry
  4. Build final prompt (history + context + question)
  5. Generate answer with Gemini
  6. Save turn to memory
"""

from app.agent import transform_query, expand_to_query_list, agentic_retrieve
from app.memory import get_or_create_session
from app.utils.prompt_builder import build_answer_prompt
from app.LLM import chat_with_gemini


async def generate_answer(
    query: str,
    session_id: str,
) -> tuple[str, list[str]]:
    """
    Parameters
    ----------
    query      : raw user question
    session_id : conversation identifier (from request)

    Returns
    -------
    answer  : generated text
    sources : list of source labels used
    """

    # 1. Conversation memory
    memory = get_or_create_session(session_id)
    history_text = memory.format_for_prompt()

    # 2. Query transformation
    transformed = await transform_query(query, history_text)
    query_variants = expand_to_query_list(transformed)
    primary_query = transformed["rewritten"]

    # 3. Agentic retrieval (hybrid → rerank → parent expand → sufficiency loop)
    context_chunks, sources = await agentic_retrieve(
        primary_query=primary_query,
        all_query_variants=query_variants,
    )

    # 4. Build prompt
    prompt = build_answer_prompt(
        query=primary_query,
        context_chunks=context_chunks,
        history_text=history_text,
    )

    # 5. Generate
    answer = await chat_with_gemini(prompt)

    # 6. Persist turn to memory
    memory.add_turn(user_message=query, assistant_message=answer)

    return answer, sources

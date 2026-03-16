"""
Prompt builder for the final answer generation step.

Combines:
  - Conversation history (if any)
  - Retrieved context chunks
  - The user's question

The prompt instructs the LLM to:
  1. Answer from the provided context only
  2. Honestly admit when the context is insufficient
  3. Cite which chunk(s) informed each claim
"""


def build_answer_prompt(
    query: str,
    context_chunks: list[dict],
    history_text: str,
) -> str:
    # Format context with numbered labels for citation
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        meta = chunk.get("metadata", {})
        file_name = meta.get("file_name", "unknown")
        page = meta.get("page")
        label = f"{file_name}, page {page}" if page else file_name
        context_parts.append(f"[Source {i+1}: {label}]\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    history_block = ""
    if history_text:
        history_block = f"""
## Conversation history
{history_text}
"""

    prompt = f"""
# Role
You are an expert research assistant. Your job is to give precise, well-grounded answers by synthesising the provided context.
{history_block}
## Retrieved context
{context_block}

## Instructions
- Answer the question using ONLY the information in the retrieved context above.
- If the context contains partial information, share what you know and clearly state what is missing.
- If the context does not contain the information needed to answer the question, respond with:
  "I don't have enough information in the knowledge base to answer this question."
  Do NOT make up facts or use outside knowledge.
- Preserve exact terminology from the source text.
- Where useful, indicate which source supports a claim, e.g. "(Source 2)".
- Be concise but complete.

## Question
{query}
""".strip()

    return prompt

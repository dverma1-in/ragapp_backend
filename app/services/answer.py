from app.schemas.chat import ChatResponse
from app.vectorstore import query_documents
from app.utils import embed_text, retrive_context_from_db
from app.utils import generate_answer_from_context


def generate_answer(query: str):
    #1st break the query into vector
    #2nd retrive context from vector store using query
    #3rd provide the llm context, prompt, query to generate answer

    query_embedding = embed_text([query])[0]
    context, sources = retrive_context_from_db(query_embedding)

    answer = generate_answer_from_context(query, context)

    return answer, sources
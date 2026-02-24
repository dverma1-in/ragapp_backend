from app.vectorstore import query_documents

def retrive_context_from_db(query_embedding):
    retrived_from_db = query_documents(query_embedding)

    documents = retrived_from_db.get("documents", [[]])[0]
    print("documents retrived from db", documents)
    metadata = retrived_from_db.get("metadatas", [[]])[0]
    print("metadata retrived from db", metadata)

    sources = []
    context_chunks = []

    for doc, meta in zip(documents, metadata):
        context_chunks.append(doc)
        
        source = meta.get("file_name", "unknown")
        page = meta.get("page")
        if page:
            sources.append(f"{source} (page {page})")
        else:
            sources.append(source)

    sources = list(dict.fromkeys(sources))    

    return "\n".join(context_chunks), sources
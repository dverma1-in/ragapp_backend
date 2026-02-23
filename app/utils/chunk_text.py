import re
from typing import List, Dict
 
def chunk_text(text: str, metadata: Dict, max_chars=600, overlap=50):
    sentences = re.split(r'(?<=[.!?])\s+', text)
 
    chunks = []
    current = ""
 
    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            chunks.append({
                "text": current.strip(),
                "metadata": metadata
            })
            current = current[-overlap:] + " " + sentence
 
    if current:
        chunks.append({
            "text": current.strip(),
            "metadata": metadata
        })
 
    return chunks
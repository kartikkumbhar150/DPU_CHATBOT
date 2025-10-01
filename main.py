import os
import time
import numpy as np
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from groq import Groq
from qdrant_client.models import QueryVector
from dotenv import load_dotenv

load_dotenv()


# ==== CONFIG ====
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==== INIT ====
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=GROQ_API_KEY)

# Memory store for last 5 QnAs
last_qna = []  # [{"q":..., "a":..., "embedding":...}]


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_related_from_memory(query_embedding, threshold=0.8):
    related_contexts = []
    for qna in last_qna:
        sim = cosine_similarity(query_embedding, qna["embedding"])
        if sim > threshold:
            related_contexts.append(f"Q: {qna['q']} A: {qna['a']}")
    return related_contexts


def query_qdrant(query, top_k=3):
    """Fetch top matches from Qdrant using the recommended query_points method."""
    query_vector_list = embed_model.encode(query).tolist()
    
    # FIX: Use qdrant_client.query_points with a QueryVector model
    hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        # Pass the vector within a QueryVector object
        query=QueryVector(vector=query_vector_list, with_vector=False), 
        limit=top_k,
        with_payload=True,
    )
    
    # Extract text from payload
    contexts = []
    # The result structure is different from the old search method
    for hit in hits.result: 
        if "text" in hit.payload:
            contexts.append(hit.payload["text"])
    return contexts

def ask_groq(question, context):
    prompt = f"""
You are a helpful assistant. Use the following context to answer.

Context from DB + memory:
{context}

Question: {question}
Answer clearly and concisely:
"""
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-120b",
        temperature=0.3,
    )
    # Fix: use .content attribute instead of dict indexing
    return response.choices[0].message.content



def chatbot(question):
    q_vec = embed_model.encode(question).tolist()

    # Step 1: get related from memory
    memory_contexts = get_related_from_memory(q_vec)

    # Step 2: fetch Qdrant context
    db_contexts = query_qdrant(question, top_k=3)

    # Step 3: build combined context
    all_context = "\n".join(memory_contexts + db_contexts)

    # Step 4: ask Groq
    answer = ask_groq(question, all_context)

    # Step 5: store in memory
    last_qna.append({"q": question, "a": answer, "embedding": q_vec})
    if len(last_qna) > 5:
        last_qna.pop(0)

    return answer


# ==== API ====
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    answer = chatbot(question)
    return jsonify({"question": question, "answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)

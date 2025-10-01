import os
import time
import numpy as np
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# ==== CONFIG & INIT (Reads from .env file) ====
# ==============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Sanity checks
if not all([QDRANT_URL, GROQ_API_KEY, COLLECTION_NAME]):
    print("FATAL: Missing one or more required environment variables (QDRANT_URL, GROQ_API_KEY, COLLECTION_NAME).")
    exit(1)

# Initialize Clients
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15)
except Exception as e:
    print(f"Error initializing Qdrant client: {e}")
    exit(1)

try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    VECTOR_SIZE = embed_model.get_sentence_embedding_dimension()
except Exception as e:
    print(f"Error initializing SentenceTransformer: {e}")
    exit(1)

# FIX: Removed the redundant 'base_url' which was causing the 404 error
# The Groq Python SDK automatically uses the correct OpenAI-compatible base URL.
try:
    groq_client = Groq(api_key=GROQ_API_KEY) # <<<<<<<<<<<<<<< FIXED HERE
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    exit(1)

# Memory store for conversational history (last 5 QnAs)
last_qna: List[Dict[str, Any]] = []


# ==============================================================================
# ==== UTILITY FUNCTIONS ====
# ==============================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)


def get_related_from_memory(query_embedding: List[float], threshold: float = 0.8) -> List[str]:
    related_contexts = []
    for qna in last_qna:
        sim = cosine_similarity(query_embedding, qna["embedding"])
        if sim > threshold:
            related_contexts.append(f"Q: {qna['q']} A: {qna['a']}")
    return related_contexts


def query_qdrant(query: str, top_k: int = 3) -> List[str]:
    """Fetch top matches from Qdrant using query_points (new API)."""
    query_vector_list = embed_model.encode(query).tolist()

    hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector_list,
        limit=top_k,
        with_payload=True,
    )

    contexts = []
    for hit in hits.points:  # query_points returns a ScoredPointList with .points
        if hit.payload and "text" in hit.payload:
            contexts.append(hit.payload["text"])
    return contexts


def ask_groq(question: str, context: str) -> str:
    prompt = f"""
You are a helpful assistant. Use ONLY the following context to answer the question.
If the context does not contain the answer, state clearly that you cannot answer based on the provided information.

Context from Database and Memory:
---
{context}
---

Question: {question}
Answer clearly and concisely:
"""
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )

        msg = response.choices[0].message
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Groq API call failed: {e}")
        return "I am currently unable to connect to the language model service."


# ==============================================================================
# ==== MAIN CHATBOT LOGIC ====
# ==============================================================================

def chatbot(question: str) -> str:
    q_vec = embed_model.encode(question).tolist()

    memory_contexts = get_related_from_memory(q_vec, threshold=0.8)
    db_contexts = query_qdrant(question, top_k=3)

    all_context = "\n".join(memory_contexts + db_contexts)

    if not all_context.strip():
        return "I'm sorry, I couldn't find any relevant information in my dedicated knowledge base to answer that question."

    answer = ask_groq(question, all_context)

    last_qna.append({"q": question, "a": answer, "embedding": q_vec})
    if len(last_qna) > 5:
        last_qna.pop(0)

    return answer


# ==============================================================================
# ==== FLASK API SETUP ====
# ==============================================================================

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
    except Exception:
        return jsonify({"error": "Invalid JSON format"}), 400

    question = data.get("question")
    if not question or not isinstance(question, str):
        return jsonify({"error": "A valid 'question' string must be provided in the request body."}), 400

    start_time = time.time()
    try:
        answer = chatbot(question)
        end_time = time.time()
        return jsonify({
            "question": question,
            "answer": answer,
            "response_time_sec": f"{(end_time - start_time):.2f}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "An internal server error occurred while processing the request.",
            "detail": str(e)
        }), 500


if __name__ == "__main__":
    print("Starting Flask RAG Chatbot Service...")

    # CHECK/CREATE COLLECTION
    try:
        try:
            qdrant_client.get_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' already exists.")
        except Exception:
            print(f"Collection '{COLLECTION_NAME}' not found. Creating with {VECTOR_SIZE} dimensions (Cosine).")
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print("Collection created successfully.")
    except Exception as e:
        print(f"Warning: Could not check/create Qdrant collection: {e}")

    app.run(host="0.0.0.0", port=8081, debug=True)
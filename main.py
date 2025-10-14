import os
import time
from typing import List, Dict, Any
import numpy as np
import traceback

from flask import Flask, request, jsonify, session
from flask_session import Session
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams, Distance

try:
    from groq import Groq
except Exception:
    Groq = None

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Tokenizer (tiktoken) for truncation safety
try:
    import tiktoken
except Exception:
    tiktoken = None

load_dotenv()

# ---- CONFIGURATION ----
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "DPU_CHATBOT").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MAX_MEMORY_ITEMS = 5
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "5800"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")


if not COLLECTION_NAME:
    print("FATAL: QDRANT_COLLECTION must be set in environment.")
    raise SystemExit(1)

# ---- INITIALIZATION ----
# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL or None, api_key=QDRANT_API_KEY or None, timeout=30)

# Embedding model (multilingual)
embed_model = SentenceTransformer(EMBED_MODEL)
VECTOR_SIZE = embed_model.get_sentence_embedding_dimension()

# Groq client
groq_client = None
if Groq and GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# ---- UTILITY FUNCTIONS ----
def truncate_to_token_limit(text: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """Safely truncates text to a specified token limit using tiktoken."""
    if tiktoken:
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens)
    else:
        # Fallback to character-based approximation if tiktoken is not available
        approx_chars = max_tokens * 3
        return text[:approx_chars]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    v1, v2 = np.array(vec1), np.array(vec2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

# ---- FLASK APP SETUP ----
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "a-secure-temporary-secret-key")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "/tmp/flask_sessions"
Session(app)

# ---- SESSION MEMORY HELPERS ----
def get_session_memory() -> List[Dict[str, Any]]:
    """Retrieves the conversation history from the current session."""
    return session.get("last_qna", [])

def save_session_memory(memory: List[Dict[str, Any]]):
    """Saves the conversation history to the current session."""
    session["last_qna"] = memory
    
def get_session_language() -> str:
    """Retrieves the preferred response language from the current session."""
    # Defaults to 'en' (English) if not set
    return session.get("preferred_lang", "en")

def get_related_from_memory(query_embedding: List[float]) -> List[str]:
    """
    Returns the entire session memory (last 5 Q&A pairs) in chronological order.
    """
    memory = get_session_memory()
    all_contexts = []
    for qna in memory:
        all_contexts.append(f"Q: {qna['q']} A: {qna['a']}")
    return all_contexts

# ---- RAG CORE LOGIC ----
def query_qdrant(query: str, top_k: int = 3) -> List[str]:
    """Queries the Qdrant vector database to find relevant documents."""
    query_vec = embed_model.encode(query).tolist()
    try:
        # Suppress DeprecationWarning from Qdrant client
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            hits = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vec,
                limit=top_k,
                with_payload=True
            )
    except Exception as e:
        print(f"Qdrant search failed: {e}")
        return []

    contexts = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or (hit.get("payload") if isinstance(hit, dict) else None)
        if payload and "text" in payload:
            contexts.append(payload["text"])
    return contexts

def ask_groq(question: str, context: str, lang: str) -> str:
    """Sends a prompt with context to the Groq LLM and gets a response."""
    if not groq_client:
        return "LLM integration not configured. Please set GROQ_API_KEY."

    prompt = f"""
You are an expert AI Assistant for Dr. D. Y. Patil Institute of Technology.
Also include Dr. D. Y. Patil Institute of Technology college name in the answer.
Your response language MUST be in '{lang}'.
You are a multilingual assistant that supports Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Malayalam, Kannada, Punjabi, Odia, etc.).
Use the following context from the knowledge base AND previous conversation to answer the question.
If the context does not contain the answer, answer it Refer to the following site https://engg.dypvp.edu.in/.
Dont include 'mentioned in the provided context documents' in the answer.
Do not repeat or restate the question in the answer.


Context from Database and Previous Turns:
---
{context}
---

Question: {question}
Answer in {lang}, clearly and concisely:
"""
    safe_prompt = truncate_to_token_limit(prompt)
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": safe_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )
        msg = response.choices[0].message
        
        return msg.content if hasattr(msg, 'content') else ""
        
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return f"Unable to connect to LLM service (error: {e})."

# ---- MAIN CHATBOT FUNCTION ----
def chatbot(question: str) -> str:
    """Orchestrates the RAG pipeline: memory, retrieval, and generation."""
    
    # Get preferred language from session. Fallback to detection if none is set.
    preferred_lang = get_session_language()
    
    if preferred_lang == "en":
        # If the preferred language is still the default ('en'), try to detect 
        # the language of the current question.
        try:
            # We don't save the detected language to session, as we want to 
            # force the user to explicitly choose a non-default language via /api/set_language
            lang = detect(question)
        except Exception:
            lang = "en"
    else:
        # Use the explicitly set preferred language for the answer
        lang = preferred_lang

    # 1. Retrieve context from session memory (ALL last 5 Q&A)
    q_vec = embed_model.encode(question).tolist()
    memory_contexts = get_related_from_memory(q_vec) 
    
    # 2. Retrieve context from vector DB (still based on relevance)
    db_contexts = query_qdrant(question, top_k=3)
    
    # 3. Combine ALL context
    all_context = "\n\n".join(memory_contexts + db_contexts)

    # 4. Handle cases with no context found
    if not all_context.strip():
        # Fallback response in the user's chosen language
        localized_fallbacks = {
            "hi": "माफ़ कीजिए, मुझे अपने ज्ञानकोष में इस प्रश्न का उत्तर नहीं मिला।",
            "ta": "மன்னிக்கவும், உங்கள் கேள்விக்கு எனது அறிவகத்தில் பதில் இல்லை.",
            "te": "క్షమించండి, మీ ప్రశ్నకు సమాధానం నా జ్ఞానంలో లభించలేదు。",
            "bn": "দুঃখিত, আমার জ্ঞানভান্ডারে আপনার প্রশ্নের উত্তর পাওয়া যায়নি।",
            "mr": "क्षमस्व, मला या प्रश्नाचे उत्तर माझ्या ज्ञानभांडारात मिळाले नाही।",
        }
        return localized_fallbacks.get(lang, "Sorry, I could not find relevant information to answer this question.")

    # 5. Generate answer using the LLM
    answer = ask_groq(question, all_context, lang) if groq_client else ("Context:\n" + all_context)

    # 6. Update session memory with the new Q&A pair (maintaining MAX_MEMORY_ITEMS)
    memory = get_session_memory()
    memory.append({"q": question, "a": answer, "embedding": q_vec})
    if len(memory) > MAX_MEMORY_ITEMS:
        memory.pop(0)
    save_session_memory(memory)

    return answer

# ---- FLASK API ENDPOINTS ----
@app.route("/api/ask", methods=["POST"])
def ask():
    """Endpoint to receive a question and return a chatbot answer."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON format"}), 400

    question = data.get("question")
    if not question or not isinstance(question, str):
        return jsonify({"error": "A valid 'question' string must be provided."}), 400

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
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.route("/api/reset", methods=["POST"])
def reset_memory():
    """Endpoint to clear the session memory AND the language preference."""
    session["last_qna"] = []
    # Also reset the preferred language
    session["preferred_lang"] = "en" 
    return jsonify({"message": "Session memory and language preference cleared successfully."})

@app.route("/api/set_language", methods=["POST"])
def set_language():
    """
    NEW ENDPOINT: Sets the preferred response language for the current session.
    The client should call this at the start of the session.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON format"}), 400

    lang_code = data.get("language_code", "").strip().lower()
    
    # Simple validation for a language code (e.g., 'en', 'hi', 'mr')
    if not lang_code or len(lang_code) not in [2, 3]:
        return jsonify({"error": "A valid 'language_code' (e.g., 'hi' for Hindi) must be provided."}), 400

    session["preferred_lang"] = lang_code
    
    # You can return a localized confirmation if desired
    confirmation_messages = {
        "hi": "नमस्ते! अब मैं आपको हिंदी में जवाब दूँगा।",
        "mr": "नमस्कार! आता मी तुम्हाला मराठीत उत्तर देईन.",
        "en": "Hello! I will now reply to you in English.",
    }
    
    message = confirmation_messages.get(lang_code, f"Language set to {lang_code}. I will now reply in this language.")
    
    return jsonify({
        "message": message,
        "language_set": lang_code
    })


# ---- APPLICATION STARTUP ----
def initialize_qdrant_collection():
    """Checks if the Qdrant collection exists and creates/recreates it if necessary."""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        current_dim = collection_info.config.params.vectors.size
        if current_dim != VECTOR_SIZE:
            print(f"Warning: Collection '{COLLECTION_NAME}' exists with dimension {current_dim}. Expected {VECTOR_SIZE}. Recreating...")
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print("Collection recreated successfully. You may need to re-run your indexing script.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists with correct dimension ({VECTOR_SIZE}).")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Creating new collection with {VECTOR_SIZE} dimensions (Cosine).")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print("Collection created successfully.")

if __name__ == "__main__":
    print("Starting Multilingual RAG Chatbot Service...")
    try:
        initialize_qdrant_collection()
    except Exception as e:
        print(f"FATAL: Could not check or create Qdrant collection: {e}")
        pass

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8081")), debug=True)
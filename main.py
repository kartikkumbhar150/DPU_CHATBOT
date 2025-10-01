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
from langdetect import detect, DetectorFactory
import tiktoken

# Ensure consistent language detection
DetectorFactory.seed = 0  

# Load environment variables
load_dotenv()

# ==============================================================================
# ==== CONFIG & INIT ====
# ==============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not all([QDRANT_URL, GROQ_API_KEY, COLLECTION_NAME]):
    print("FATAL: Missing one or more required environment variables.")
    exit(1)

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15)

# Use a multilingual embedding model (supports Indian languages)
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
VECTOR_SIZE = embed_model.get_sentence_embedding_dimension()

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Memory store (last 5 interactions)
last_qna: List[Dict[str, Any]] = []

# Tokenizer setup (OpenAI/Groq compatible)
MAX_TOKENS = 5800
try:
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
except Exception:
    tokenizer = tiktoken.get_encoding("cl100k_base")

# ==============================================================================
# ==== UTILITY FUNCTIONS ====
# ==============================================================================

def truncate_to_token_limit(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Ensure text does not exceed token limit."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

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
    query_vector_list = embed_model.encode(query).tolist()
    hits = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector_list,
        limit=top_k,
        with_payload=True,
    )
    contexts = []
    for hit in hits.points:
        if hit.payload and "text" in hit.payload:
            contexts.append(hit.payload["text"])
    return contexts

def ask_groq(question: str, context: str, lang: str) -> str:
    prompt = f"""
You are a multilingual assistant that supports Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Malayalam, Kannada, Punjabi, Odia, etc.).
Use ONLY the following context to answer the question.
If the context does not contain the answer, clearly say you cannot answer from the given data.

Context from Database and Memory:
---
{context}
---

Question ({lang}): {question}
Answer in {lang}, clearly and concisely:
"""
    # Truncate to 5800 tokens before sending
    safe_prompt = truncate_to_token_limit(prompt, MAX_TOKENS)

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": safe_prompt}],
            model="llama-3.1-8b-instant",  # better multilingual support
            temperature=0.3,
        )
        msg = response.choices[0].message
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "")
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ Unable to connect to LLM service (error: {e})."

# ==============================================================================
# ==== MAIN CHATBOT LOGIC ====
# ==============================================================================

def chatbot(question: str) -> str:
    try:
        lang = detect(question)  # auto-detect language
    except Exception:
        lang = "English"

    q_vec = embed_model.encode(question).tolist()
    memory_contexts = get_related_from_memory(q_vec, threshold=0.8)
    db_contexts = query_qdrant(question, top_k=3)
    all_context = "\n".join(memory_contexts + db_contexts)

    if not all_context.strip():
        localized_fallbacks = {
            "hi": "माफ़ कीजिए, मुझे अपने ज्ञानकोष में इस प्रश्न का उत्तर नहीं मिला।",
            "ta": "மன்னிக்கவும், உங்கள் கேள்விக்கு எனது அறிவகத்தில் பதில் இல்லை.",
            "te": "క్షమించండి, మీ ప్రశ్నకు సమాధానం నా జ్ఞానంలో లభించలేదు.",
            "bn": "দুঃখিত, আমার জ্ঞানভান্ডারে আপনার প্রশ্নের উত্তর পাওয়া যায়নি।",
            "mr": "क्षमस्व, मला या प्रश्नाचे उत्तर माझ्या ज्ञानभांडारात मिळाले नाही.",
            "gu": "માફ કરશો, તમારા પ્રશ્નનો જવાબ મારા જ્ઞાનભંડારમાં મળ્યો નથી.",
            "ml": "ക്ഷമിക്കണം, നിങ്ങളുടെ ചോദ്യത്തിന് എന്റെ അറിവില്‍ ഉത്തരമില്ല.",
            "kn": "ಕ್ಷಮಿಸಿ, ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನನ್ನ ಜ್ಞಾನಕೋಶದಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ.",
            "pa": "ਮਾਫ ਕਰਨਾ, ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਮੇਰੇ ਗਿਆਨ ਵਿੱਚ ਨਹੀਂ ਮਿਲਿਆ।",
            "or": "ମାପ କରନ୍ତୁ, ଆମର ଜ୍ଞାନଭଣ୍ଡାରରେ ଆପଣଙ୍କ ପ୍ରଶ୍ନର ଉତ୍ତର ମିଳିଲା ନାହିଁ।",
        }
        return localized_fallbacks.get(lang, "Sorry, I could not find relevant information to answer this question.")

    answer = ask_groq(question, all_context, lang)
    last_qna.append({"q": question, "a": answer, "embedding": q_vec})
    if len(last_qna) > 5:
        last_qna.pop(0)
    return answer

# ==============================================================================
# ==== FLASK API ====
# ==============================================================================

app = Flask(__name__)

@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        data = request.json
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
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "detail": str(e)
        }), 500

if __name__ == "__main__":
    print("Starting Multilingual RAG Chatbot Service...")

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

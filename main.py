import os
import time
import re
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
# Set seed for reproducibility in language detection
DetectorFactory.seed = 0 

# Tokenizer (tiktoken) for truncation safety
try:
    import tiktoken
except Exception:
    tiktoken = None

load_dotenv()

# ---- CONFIGURATION ----
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    SESSION_TYPE = "filesystem"
    SESSION_FILE_DIR = "/tmp/flask_sessions"
    MAX_MEMORY_ITEMS = int(os.getenv("MAX_MEMORY_ITEMS", "5"))
    MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "5800"))
    EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
    
    QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "DPU_CHATBOT").strip()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    
    SUPPORTED_LANGUAGES = ["en", "hi", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "or"]
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.COLLECTION_NAME:
            raise ValueError("QDRANT_COLLECTION must be set in environment")
        
        if not cls.SECRET_KEY:
            if os.getenv("FLASK_ENV") == "production":
                raise ValueError("SECRET_KEY must be set in production environment")
            else:
                cls.SECRET_KEY = "dev-key-do-not-use-in-production"
                print("Warning: Using development SECRET_KEY")

# Validate configuration
try:
    Config.validate()
except ValueError as e:
    print(f"FATAL: Configuration error: {e}")
    raise SystemExit(1)

# ---- INITIALIZATION ----
print(f"Initializing Qdrant Client for collection: {Config.COLLECTION_NAME}")
# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=Config.QDRANT_URL or None, 
    api_key=Config.QDRANT_API_KEY or None, 
    timeout=30
)

# Embedding model (multilingual)
print(f"Loading Sentence Transformer model: {Config.EMBED_MODEL}")
embed_model = SentenceTransformer(Config.EMBED_MODEL)
VECTOR_SIZE = embed_model.get_sentence_embedding_dimension()

# Groq client
groq_client = None
if Groq and Config.GROQ_API_KEY:
    print("Initializing Groq Client.")
    groq_client = Groq(api_key=Config.GROQ_API_KEY)
else:
    print("Warning: GROQ_API_KEY not set. LLM functionality disabled.")

# ---- FLASK APP SETUP ----
app = Flask(__name__)
app.config.from_object(Config)
Session(app)

# ---- UTILITY FUNCTIONS ----
def sanitize_input(text: str) -> str:
    """Basic input sanitization"""
    if not text or not isinstance(text, str):
        return ""
    # Remove excessive whitespace and potentially dangerous characters
    text = re.sub(r'[^\w\s\?\.\,\!\-@#$%&*()]', '', text.strip())
    return text[:1000]  # Limit input length

def truncate_to_token_limit(text: str, max_tokens: int = Config.MAX_PROMPT_TOKENS) -> str:
    """Safely truncates text to a specified token limit using tiktoken."""
    if tiktoken:
        try:
            # Prefer a common tokenizer model
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            # Fallback to base encoding
            tokenizer = tiktoken.get_encoding("cl100k_base")
        
        tokens = tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens)
    else:
        # Fallback to rough character count if tiktoken is not available
        approx_chars = max_tokens * 3
        return text[:approx_chars]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    v1, v2 = np.array(vec1), np.array(vec2)
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

def get_response_language(question: str, preferred_lang: str) -> str:
    """More robust language detection"""
    if preferred_lang != "en":
        return preferred_lang
        
    try:
        # Use multiple detections for better accuracy
        DetectorFactory.seed = 0
        lang = detect(question)
        # Validate it's a supported language
        return lang if lang in Config.SUPPORTED_LANGUAGES else "en"
    except Exception:
        return "en"

# ---- SESSION MEMORY HELPERS (Centralized History Management) ----
def get_session_memory() -> List[Dict[str, Any]]:
    """Retrieves the conversation history from the current session."""
    # This stores up to MAX_MEMORY_ITEMS (5) Q&A pairs
    return session.get("last_qna", [])

def save_session_memory(memory: List[Dict[str, Any]]):
    """Saves the conversation history to the current session."""
    session["last_qna"] = memory

def get_session_language() -> str:
    """Retrieves the preferred response language from the current session."""
    return session.get("preferred_lang", "en")

def get_related_from_memory(query_embedding: List[float]) -> List[str]:
    """
    Returns the chronological session memory (last 5 Q&A pairs) 
    formatted for the LLM context.
    """
    memory = get_session_memory()
    all_contexts = []
    # The memory is already limited to MAX_MEMORY_ITEMS (5) in the chatbot function.
    for qna in memory:
        # Format as Q: ... \n A: ... for clear context separation
        all_contexts.append(f"Q: {qna['q']}\nA: {qna['a']}")
    
    return all_contexts

# ---- RAG CORE LOGIC ----
def query_qdrant(query: str, top_k: int = 3) -> List[str]:
    """Queries the Qdrant vector database to find relevant documents."""
    query_vec = embed_model.encode(query).tolist()
    try:
        # Suppress the DeprecationWarning from Qdrant client's internals
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            hits = qdrant_client.search(
                collection_name=Config.COLLECTION_NAME,
                query_vector=query_vec,
                limit=top_k,
                with_payload=True
            )
    except Exception as e:
        print(f"Qdrant search failed: {e}")
        # Log the full traceback for debugging the Qdrant connection
        traceback.print_exc() 
        return []

    contexts = []
    for hit in hits:
        # Handle both dict-like and object-like hit structures
        payload = getattr(hit, "payload", None) or (hit.get("payload") if isinstance(hit, dict) else None)
        if payload and "text" in payload:
            contexts.append(payload["text"])
    return contexts

def ask_groq(question: str, context: str, lang: str) -> str:
    """Sends a prompt with context to the Groq LLM and gets a response."""
    if not groq_client:
        return "LLM integration not configured. Please set GROQ_API_KEY."

    script_requirements = {
        "hi": "Devanagari script",
        "mr": "Devanagari script", 
        "bn": "Bengali script",
        "ta": "Tamil script",
        "te": "Telugu script",
        "kn": "Kannada script",
        "ml": "Malayalam script",
        "gu": "Gujarati script",
        "pa": "Gurmukhi script",
        "or": "Odia script",
        "en": "English script"
    }
    script = script_requirements.get(lang, "native script")


    prompt = f"""
You are an expert AI Assistant for Dr. D. Y. Patil Institute of Technology.
Also include Dr. D. Y. Patil Institute of Technology college name in the answer.
Your response language MUST be in '{lang}'. 
Your MUST write the response in '{script}'. 
Examples of correct script usage:
- For Marathi (mr): Use Devanagari script - "डॉ. डी. वाय. पाटील इंस्टिट्यूट ऑफ टेक्नॉलॉजी"
- For Hindi (hi): Use Devanagari script - "डॉ. डी. वाय. पाटील प्रौद्योगिकी संस्थान"
- For Tamil (ta): Use Tamil script - "டாக்டர். டி. ஒய். பாட்டில் தொழில்நுட்ப நிறுவனம்"
- For Bengali (bn): Use Bengali script - "ডক্টর. ডি. ওয়াই. পাটিল ইনস্টিটিউট অফ টেকনোলজি"
- For Telugu (te): Use Telugu script - "డాక్టర్. డి. వై. పటిల్ ఇన్స్టిట్యూట్ ఆఫ్ టెక్నాలజీ"
- For Kannada (kn): Use Kannada script - "ಡಾ. ಡಿ. ವೈ. ಪಾಟೀಲ್ ಇನ್ಸ್ಟಿಟ್ಯೂಟ್ ಆಫ್ ಟೆಕ್ನಾಲಜಿ"
- For Malayalam (ml): Use Malayalam script - "ഡോ. ഡി. വൈ. പാട്ടീൽ ഇൻസ്റ്റിറ്റ്യൂട്ട് ഓഫ് ടെക്നോളജി"
- For Gujarati (gu): Use Gujarati script - "ડૉ. ડી. વાઈ. પાટીલ ઇન્સ્ટિટ્યૂટ ઓફ ટેકનોલોજી"
- For Punjabi (pa): Use Gurmukhi script - "ਡਾ. ਡੀ. ਵਾਈ. ਪਾਟੀਲ ਇੰਸਟੀਚਿਊਟ ਆਫ਼ ਟੈਕਨੋਲੋਜੀ"
- For Odia (or): Use Odia script - "ଡା. ଡି. ୱାଇ. ପାଟିଲ୍ ଇନ୍‌ସ୍ଟିଚ୍ୟୁଟ୍ ଅଫ୍ ଟେକ୍ନୋଲୋଜି" 
You are a multilingual assistant that supports Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Malayalam, Kannada, Punjabi, Odia, etc.).
Use the following context from the knowledge base AND previous conversation to answer the question.
If the context does not contain the answer, answer it Refer to the following site https://engg.dypvp.edu.in/.
Dont include 'mentioned in the provided context documents' in the answer.
Do not repeat or restate the question in the answer.
- Never use Latin alphabet transliteration for Indian languages
- Always use the proper native script characters

Context from Database and Session Memory:
---
{context}
---

Question: {question}

Answer in {lang} using {script} (native characters only):
"""
    # Truncate the entire prompt safely
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
        traceback.print_exc()
        return f"Unable to connect to LLM service (error: {e})."

# ---- MAIN CHATBOT FUNCTION ----
def chatbot(question: str) -> str:
    """Orchestrates the RAG pipeline: memory, retrieval, and generation."""

    preferred_lang = get_session_language()
    lang = get_response_language(question, preferred_lang)

    # --- 1. CONVERSATION HISTORY CONTEXT (Session Memory) ---
    q_vec = embed_model.encode(question).tolist()
    # This retrieves the last 5 Q&A pairs, formatted for context.
    memory_contexts = get_related_from_memory(q_vec) 

    # --- 2. VECTOR DB CONTEXT (RAG Retrieval) ---
    db_contexts = query_qdrant(question, top_k=3)

    # --- 3. COMBINE CONTEXTS ---
    # Combining Session Memory (history) and Database Context (knowledge)
    all_context = "\n\n".join(memory_contexts + db_contexts)

    if not all_context.strip():
        # Fallback if no information is found anywhere
        localized_fallbacks = {
            "hi": "माफ़ कीजिए, मुझे अपने ज्ञानकोष या बातचीत के इतिहास में इस प्रश्न का उत्तर नहीं मिला।",
            "ta": "மன்னிக்கவும், உங்கள் கேள்விக்கு எனது அறிவகத்திலோ அல்லது உரையாடல் வரலாற்றிலோ பதில் இல்லை.",
            "te": "క్షమించండి, మీ ప్రశ్నకు సమాధానం నా జ్ఞానంలో లేదా సంభాషణ చరిత్రలో లభించలేదు。",
            "bn": "দুঃখিত, আমার জ্ঞানভান্ডার বা আলোচনার ইতিহাসে আপনার প্রশ্নের উত্তর পাওয়া যায়নি।",
            "mr": "क्षमस्व, मला या प्रश्नाचे उत्तर माझ्या ज्ञानभांडारात किंवा संभाषणाच्या इतिहासात मिळाले नाही।",
            "gu": "માફ કરશો, મને તમારા પ્રશ્નનો જવાબ મારા જ્ઞાનકોશ અથવા વાર્તાલાપ ઇતિહાસમાં નથી મળ્યો.",
            "kn": "ಕ್ಷಮಿಸಿ, ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನನ್ನ ಜ್ಞಾನಕೋಶ ಅಥವಾ ಸಂಭಾಷಣೆ ಇತಿಹಾಸದಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ.",
            "ml": "ക്ഷമിക്കണം, എന്റെ നോളഡ്ജ് ബേസ് അല്ലെങ്കിൽ സംഭാഷണ ചരിത്രത്തിൽ നിങ്ങളുടെ ചോദ്യത്തിന് ഉത്തരം കണ്ടെത്തിയില്ല.",
            "pa": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਨੂੰ ਤੁਹਾਡੇ ਸਵਾਲ ਦਾ ਜਵਾਬ ਆਪਣੇ ਨਾਲ਼ਜ ਬੇਸ ਜਾਂ ਗੱਲਬਾਤ ਦੇ ਇਤਿਹਾਸ ਵਿੱਚ ਨਹੀਂ ਮਿਲਿਆ।",
            "or": "ଦୁଃଖିତ, ମୁଁ ଆପଣଙ୍କ ପ୍ରଶ୍ନର ଉତ୍ତର ମୋର ଜ୍ଞାନଭଣ୍ଡାର କିମ୍ବା କଥୋପକଥନ ଇତିହାସରେ ପାଇଲି ନାହିଁ।",
        }
        return localized_fallbacks.get(lang, "Sorry, I could not find relevant information in my knowledge base or history to answer this question.")

    # --- 4. ASK GROQ LLM ---
    answer = ask_groq(question, all_context, lang) if groq_client else ("Context:\n" + all_context)

    # --- 5. UPDATE SESSION MEMORY ---
    # Only update the Session memory
    memory = get_session_memory()
    memory.append({"q": question, "a": answer, "embedding": q_vec})
    
    # Enforce MAX_MEMORY_ITEMS limit (FIFO: pop the oldest item)
    if len(memory) > Config.MAX_MEMORY_ITEMS:
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

    question = sanitize_input(data.get("question", ""))
    if not question:
        return jsonify({"error": "A valid non-empty 'question' must be provided."}), 400

    start_time = time.time()
    try:
        answer = chatbot(question)
        end_time = time.time()
        return jsonify({
            "question": question,
            "answer": answer,
            "response_time_sec": f"{(end_time - start_time):.2f}",
            "language_used": get_session_language()
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.route("/api/reset", methods=["POST"])
def reset_memory():
    """Endpoint to clear the session memory and language preference."""
    session["last_qna"] = []
    session["preferred_lang"] = "en"
    return jsonify({"message": "Session memory and language preference cleared successfully."})

@app.route("/api/set_language", methods=["POST"])
def set_language():
    """Sets the preferred response language for the current session."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON format"}), 400

    lang_code = data.get("language_code", "").strip().lower()
    if not lang_code or lang_code not in Config.SUPPORTED_LANGUAGES:
        return jsonify({
            "error": f"A valid 'language_code' must be provided. Supported: {', '.join(Config.SUPPORTED_LANGUAGES)}"
        }), 400

    session["preferred_lang"] = lang_code

    confirmation_messages = {
        "hi": "नमस्ते! अब मैं आपको हिंदी में जवाब दूँगा।",
        "mr": "नमस्कार! आता मी तुम्हाला मराठीत उत्तर देईन.",
        "ta": "வணக்கம்! இனி நான் தமிழில் பதிலளிப்பேன்.",
        "te": "నమస్కారం! ఇకపై నేను తెలుగులో జవాబు ఇస్తాను.",
        "bn": "নমস্কার! এখন থেকে আমি বাংলায় উত্তর দেব।",
        "gu": "નમસ્તે! હવે હું તમને ગુજરાતીમાં જવાબ આપીશ.",
        "kn": "ನಮಸ್ಕಾರ! ಇನ್ನು ಮುಂದೆ ನಾನು ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರ ನೀಡುತ್ತೇನೆ.",
        "ml": "നമസ്കാരം! ഇനി ഞാൻ മലയാളത്തിൽ മറുപടി നൽകും.",
        "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਹੁਣ ਮੈਂ ਤੁਹਾਨੂੰ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਆਂਗਾ।",
        "or": "ନମସ୍କାର! ଏବେ ମୁଁ ଆପଣଙ୍କୁ ଓଡ଼ିଆରେ ଉତ୍ତର ଦେବି।",
        "en": "Hello! I will now reply to you in English.",
    }

    message = confirmation_messages.get(lang_code, f"Language set to {lang_code}. I will now reply in this language.")

    return jsonify({
        "message": message,
        "language_set": lang_code
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "qdrant": "unknown",
            "groq": "unknown", 
            "embedding_model": "unknown"
        }
    }
    
    # Check Qdrant
    try:
        qdrant_client.get_collections()
        status["services"]["qdrant"] = "healthy"
    except Exception as e:
        status["services"]["qdrant"] = "unhealthy"
        status["qdrant_error"] = str(e)
    
    # Check Groq
    if groq_client:
        try:
            # Simple test call to verify Groq connectivity
            groq_client.models.list()
            status["services"]["groq"] = "healthy"
        except Exception as e:
            status["services"]["groq"] = "unhealthy"
            status["groq_error"] = str(e)
    else:
        status["services"]["groq"] = "disabled"
    
    # Check embedding model
    try:
        # Simple test encoding
        embed_model.encode("test")
        status["services"]["embedding_model"] = "healthy"
    except Exception as e:
        status["services"]["embedding_model"] = "unhealthy"
        status["embedding_error"] = str(e)
    
    # Overall status
    unhealthy_services = [svc for svc, stat in status["services"].items() if stat != "healthy"]
    if unhealthy_services and status["services"]["qdrant"] != "healthy":
        status["status"] = "unhealthy"
    
    return jsonify(status)

@app.route("/api/supported_languages", methods=["GET"])
def supported_languages():
    """Returns the list of supported languages"""
    return jsonify({
        "supported_languages": Config.SUPPORTED_LANGUAGES,
        "default_language": "en"
    })

# ---- ERROR HANDLERS ----
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

# ---- APPLICATION STARTUP ----
def initialize_qdrant_collection():
    """Improved collection initialization with better error handling"""
    if not Config.QDRANT_URL and not Config.QDRANT_API_KEY:
        print("Warning: QDRANT credentials not set. Skipping collection check.")
        return
        
    try:
        collections = qdrant_client.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        if Config.COLLECTION_NAME in existing_collections:
            collection_info = qdrant_client.get_collection(Config.COLLECTION_NAME)
            current_dim = collection_info.config.params.vectors.size
            if current_dim != VECTOR_SIZE:
                print(f"Dimension mismatch ({current_dim} vs {VECTOR_SIZE}). Recreating collection...")
                qdrant_client.recreate_collection(
                    collection_name=Config.COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                )
                print("Collection recreated with correct dimensions.")
            else:
                print(f"Collection exists with correct dimensions ({VECTOR_SIZE}).")
        else:
            # Create new collection
            qdrant_client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print("Collection created successfully.")
            
    except Exception as e:
        print(f"Error initializing Qdrant collection: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Multilingual RAG Chatbot Service...")
    try:
        # Check and initialize Qdrant before starting the server
        initialize_qdrant_collection() 
    except SystemExit:
        # Exit if configuration validation fails
        pass 
    except Exception as e:
        print(f"FATAL: Qdrant setup failed. Service may be limited: {e}")
        traceback.print_exc()

    print("Web server running...")
    # Using '0.0.0.0' for deployment flexibility
    app.run(
        host="0.0.0.0", 
        port=int(os.getenv("PORT", "8081")), 
        debug=os.getenv("FLASK_ENV") != "production"
    )
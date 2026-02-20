"""
Phase IV: FastAPI Backend for Urdu Story Generation
- POST /generate - Generate story with streaming
- POST /signup - Create user account
- POST /login - Authenticate and get JWT
- GET /chats - Get user's saved chats
- POST /chats - Save a chat
- DELETE /chats/{id} - Delete a chat
- Serves React frontend static files
"""

import os
import json
import time
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
import jwt
import asyncio

from bpe_tokenizer import BPETokenizer
from ngram_model import NGramModel, get_model_filename

# Global cache for models and tokenizers
# Key: (n, vocab_size), Value: (model, tokenizer)
model_cache = {}

def get_model_and_tokenizer(n, vocab_size):
    key = (n, vocab_size)
    if key in model_cache:
        return model_cache[key]
    
    print(f"Loading {n}-gram model with {vocab_size} vocab...")
    t = BPETokenizer()
    try:
        t.load(vocab_size=vocab_size)
    except Exception as e:
        print(f"Error loading tokenizer for v{vocab_size}: {e}. Falling back to 250.")
        t.load(vocab_size=250)
        vocab_size = 250
        key = (n, vocab_size)
        if key in model_cache: return model_cache[key]

    m = NGramModel(n=n)
    model_file = get_model_filename(n, vocab_size)
    
    # Fallback to trigram_model.json if default n=3, v=250 is requested and new file doesn't exist
    if n == 3 and vocab_size == 250 and not os.path.exists(model_file):
        if os.path.exists('trigram_model.json'):
            model_file = 'trigram_model.json'

    try:
        m.load(model_file)
    except Exception as e:
        print(f"Error loading model {model_file}: {e}. Falling back to default trigram.")
        if n != 3 or vocab_size != 250:
            return get_model_and_tokenizer(3, 250)
        raise e

    model_cache[key] = (m, t)
    return m, t

# Configuration
# Priority: APP_SECRET_KEY env-var (production/Docker) → .secret_key file (dev)
_KEY_FILE = ".secret_key"
SECRET_KEY = os.environ.get("APP_SECRET_KEY", "").strip()
if not SECRET_KEY:
    if os.path.exists(_KEY_FILE):
        with open(_KEY_FILE) as _f:
            SECRET_KEY = _f.read().strip()
    else:
        SECRET_KEY = secrets.token_hex(32)
        with open(_KEY_FILE, "w") as _f:
            _f.write(SECRET_KEY)

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
DB_PATH = "app_data.db"

# Global references removed in favor of model_cache


def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        messages TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()


def hash_password(password):
    """Hash password with salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{hashed.hex()}"


def verify_password(password, stored_hash):
    """Verify password against stored hash."""
    salt, hashed = stored_hash.split(':')
    check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return check.hex() == hashed


def create_token(user_id, username):
    """Create JWT token."""
    payload = {
        'user_id': user_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(request: Request):
    """Extract user from JWT token in Authorization header."""
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth[7:]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {'user_id': payload['user_id'], 'username': payload['username']}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Default Model (Trigram, 250)...")
    try:
        get_model_and_tokenizer(3, 250)
    except Exception as e:
        print(f"Warning: Default model not found. You may need to train it. Error: {e}")
    
    init_db()
    yield
    print("Server shutting down...")


app = FastAPI(title="Urdu Story Generator API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Health Check -----

@app.get("/health", tags=["ops"],
         summary="Liveness probe",
         response_description="Service and model status")
async def health():
    """Used by Docker HEALTHCHECK and CI smoke tests."""
    return {
        "status": "ok",
        "models_cached": list(model_cache.keys()),
    }


# ----- Pydantic Models -----

from typing import Optional, Annotated
from pydantic import Field

class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class GenerateRequest(BaseModel):
    """POST /generate — inference request."""
    prefix: Annotated[str, Field(default="", max_length=500,
                                  description="Urdu text prefix to continue (max 500 chars)")]
    max_length: Annotated[int, Field(default=100, ge=10, le=500,
                                      description="Number of tokens to generate (10–500)")]
    temperature: Annotated[float, Field(default=0.7, ge=0.1, le=2.0,
                                        description="Sampling temperature (0.1–2.0)")]
    repetition_penalty: Annotated[float, Field(default=1.3, ge=1.0, le=3.0,
                                               description="Repetition penalty (1.0–3.0)")]
    n_gram: Annotated[int, Field(default=3, description="N-gram order (3, 5, 7)")]
    vocab_size: Annotated[int, Field(default=250, description="Vocab size (250, 500, 1000, 5000)")]

class SaveChatRequest(BaseModel):
    id: Optional[int] = None  # Optional ID to update existing chat
    title: str
    messages: str  # JSON string



# ----- Auth Endpoints -----

@app.post("/signup")
async def signup(req: SignupRequest):
    if len(req.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(req.password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                  (req.username, hash_password(req.password)))
        conn.commit()
        user_id = c.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(400, "Username already exists")
    conn.close()
    
    token = create_token(user_id, req.username)
    return {"message": "Account created", "token": token, "username": req.username}


@app.post("/login")
async def login(req: LoginRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username = ?", (req.username,))
    row = c.fetchone()
    conn.close()
    
    if not row or not verify_password(req.password, row[1]):
        raise HTTPException(401, "Invalid username or password")
    
    token = create_token(row[0], req.username)
    return {"token": token, "username": req.username}


# ----- Generate Endpoint -----

@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate story with streaming response."""
    try:
        model, tokenizer = get_model_and_tokenizer(req.n_gram, req.vocab_size)
    except Exception as e:
        raise HTTPException(503, f"Model not available: {str(e)}")
    
    max_len = min(req.max_length, 500)
    
    async def stream_tokens():
        prev_text = ""
        for text in model.generate_streaming(
            tokenizer,
            prefix=req.prefix,
            max_length=max_len,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty
        ):
            # Clean special characters from output for display
            clean_text = text.replace('\u0600', '').replace('\u0601', '\n\n').replace('\u0602', '')
            # Send only new content
            new_content = clean_text[len(prev_text):]
            if new_content:
                yield f"data: {json.dumps({'text': new_content, 'full_text': clean_text}, ensure_ascii=False)}\n\n"
                prev_text = clean_text
            await asyncio.sleep(0.01)  # Faster delay
        yield f"data: {json.dumps({'done': True, 'full_text': prev_text}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(stream_tokens(), media_type="text/event-stream")


@app.post("/generate-sync")
async def generate_sync(req: GenerateRequest):
    """Generate story synchronously (non-streaming)."""
    try:
        model, tokenizer = get_model_and_tokenizer(req.n_gram, req.vocab_size)
    except Exception as e:
        raise HTTPException(503, f"Model not available: {str(e)}")
    
    max_len = min(req.max_length, 500)
    result = model.generate(
        tokenizer,
        prefix=req.prefix,
        max_length=max_len,
        temperature=req.temperature,
        repetition_penalty=req.repetition_penalty
    )
    # Clean special characters
    clean_result = result.replace('\u0600', '').replace('\u0601', '\n\n').replace('\u0602', '')
    return {"generated_text": clean_result}


# ----- Chat History Endpoints -----

@app.get("/chats")
async def get_chats(request: Request):
    user = get_current_user(request)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, messages, created_at, updated_at FROM chats WHERE user_id = ? ORDER BY updated_at DESC",
              (user['user_id'],))
    rows = c.fetchall()
    conn.close()
    
    return [{"id": r[0], "title": r[1], "messages": r[2], "created_at": r[3], "updated_at": r[4]} for r in rows]


@app.post("/chats")
async def save_chat(req: SaveChatRequest, request: Request):
    user = get_current_user(request)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if req.id:
        # Update existing chat
        c.execute("UPDATE chats SET messages = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                  (req.messages, req.id, user['user_id']))
        chat_id = req.id
    else:
        # Create new chat
        c.execute("INSERT INTO chats (user_id, title, messages) VALUES (?, ?, ?)",
                  (user['user_id'], req.title, req.messages))
        chat_id = c.lastrowid
        
    conn.commit()
    conn.close()
    return {"id": chat_id, "message": "Chat saved"}


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int, request: Request):
    user = get_current_user(request)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user['user_id']))
    conn.commit()
    deleted = c.rowcount
    conn.close()
    if deleted == 0:
        raise HTTPException(404, "Chat not found")
    return {"message": "Chat deleted"}


# ----- Serve Frontend -----

# Serve the self-contained CDN React index.html
@app.get("/")
async def serve_frontend():
    # Use absolute path relative to this file for robustness
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(current_dir, "frontend", "index.html")
    
    if os.path.exists(frontend_path):
        return FileResponse(
            frontend_path, 
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    
    # Fallback to current directory index.html (Docker structure)
    local_index = os.path.join(current_dir, "index.html")
    if os.path.exists(local_index):
        return FileResponse(local_index, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    return HTMLResponse("<h1>Urdu Story Generator API</h1><p>Frontend not found. Error 404.</p>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
    except Exception as e:
        print(f"Uvicorn error: {e}")

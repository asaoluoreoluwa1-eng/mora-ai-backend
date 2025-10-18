#!/usr/bin/env python3
"""
MORA multilingual backend (single file)
- Chat with language support (auto-detect or user-specified)
- /speak -> returns MP3 audio (gTTS fallback)
- /transcribe -> accepts audio file and uses OpenAI Whisper if available
- Keeps auth, file-pin, memory, RAG and streaming chat behavior
Run:
  pip install -r requirements.txt
  export OPENAI_API_KEY=sk-...
  export JARVIS_JWT_SECRET=yoursecret
  uvicorn mora_multilang_server:app --host 0.0.0.0 --port 8000
"""
import os, uuid, sqlite3, json, datetime, time
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# Config
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
JWT_SECRET = os.getenv("JARVIS_JWT_SECRET", "change-this-secret")
OPENAI_AVAILABLE = bool(OPENAI_KEY)

# Optional libs
try: import openai
except: openai = None
try: import pandas as pd
except: pd = None
try: import fitz as pymupdf
except: pymupdf = None
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBER_AVAILABLE = True
except:
    SentenceTransformer = None; cosine_similarity = None; SBER_AVAILABLE = False

# TTS library (gTTS supports many languages)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except:
    GTTS_AVAILABLE = False

# Language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except:
    LANGDETECT_AVAILABLE = False

# Auth libs
try:
    from passlib.context import CryptContext
    from jose import jwt, JWTError
    PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")
    AUTH_AVAILABLE = True
except:
    PWD_CTX = None; jwt = None; JWTError = Exception; AUTH_AVAILABLE = False

# Paths & DB
BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "uploads"; UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE / "mora_multilang.db"

# ---- Tiny DB (users/files/chats/memories) ----
class DB:
    def __init__(self,path=DB_PATH):
        self.conn = sqlite3.connect(str(path),check_same_thread=False)
        self.c = self.conn.cursor()
        self._ensure()
    def _ensure(self):
        self.c.execute("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, email TEXT UNIQUE, phone TEXT UNIQUE, password TEXT, created_at TEXT)""")
        self.c.execute("""CREATE TABLE IF NOT EXISTS files(id TEXT PRIMARY KEY, user_id INTEGER, name TEXT, path TEXT, mime TEXT, created_at TEXT)""")
        self.c.execute("""CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY, user_id INTEGER, role TEXT, text TEXT, ts TEXT)""")
        self.c.execute("""CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, user_id INTEGER, text TEXT, embedding_json TEXT, created_at TEXT)""")
        self.conn.commit()
    # users
    def create_user(self, username, email, phone, password_hash):
        ts = datetime.datetime.datetime.utcnow().isoformat()
        self.c.execute("INSERT INTO users(username,email,phone,password,created_at) VALUES(?,?,?,?,?)",(username,email,phone,password_hash,datetime.datetime.datetime.utcnow().isoformat()))
        self.conn.commit()
    def get_user_by_username(self, username):
        return self.c.execute("SELECT id,username,email,phone,password FROM users WHERE username=?", (username,)).fetchone()
    def get_user_by_email(self, email):
        return self.c.execute("SELECT id,username,email,phone,password FROM users WHERE email=?", (email,)).fetchone()
    def get_user_by_phone(self, phone):
        return self.c.execute("SELECT id,username,email,phone,password FROM users WHERE phone=?", (phone,)).fetchone()
    # files
    def add_file(self, user_id, fid, name, path, mime):
        self.c.execute("INSERT INTO files(id,user_id,name,path,mime,created_at) VALUES(?,?,?,?,?,?)",(fid,user_id,name,str(path),mime,datetime.datetime.datetime.utcnow().isoformat()))
        self.conn.commit()
    def list_files(self, user_id):
        return [dict(id=r[0],name=r[2],mime=r[4]) for r in self.c.execute("SELECT * FROM files WHERE user_id=? ORDER BY rowid DESC",(user_id,)).fetchall()]
    def get_file(self,fid):
        return self.c.execute("SELECT id,user_id,name,path,mime FROM files WHERE id=?",(fid,)).fetchone()
    # chats
    def add_chat(self,user_id,role,text):
        self.c.execute("INSERT INTO chats(user_id,role,text,ts) VALUES(?,?,?,?)",(user_id,role,text,datetime.datetime.datetime.utcnow().isoformat()))
        self.conn.commit()
    def recent_chats(self,user_id,limit=20):
        rows = self.c.execute("SELECT role,text,ts FROM chats WHERE user_id=? ORDER BY id DESC LIMIT ?",(user_id,limit)).fetchall()
        return list(reversed(rows))
    # memories
    def add_memory(self,user_id,text,embedding_json=None):
        self.c.execute("INSERT INTO memories(user_id,text,embedding_json,created_at) VALUES(?,?,?,?)",(user_id,text,embedding_json or "",datetime.datetime.datetime.utcnow().isoformat()))
        self.conn.commit()
    def list_memories(self,user_id,limit=200):
        return self.c.execute("SELECT id,text,embedding_json FROM memories WHERE user_id=? ORDER BY id DESC LIMIT ?",(user_id,limit)).fetchall()

DB = DB()

# ---- embeddings (OpenAI preferred, SBERT fallback) ----
EMBED_MODEL = None
if SBER_AVAILABLE:
    try:
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except:
        EMBED_MODEL = None

def get_openai_embedding(text):
    if not OPENAI_AVAILABLE or openai is None:
        return None
    openai.api_key = OPENAI_KEY
    try:
        res = openai.Embedding.create(model="text-embedding-3-small", input=text)
        return res['data'][0]['embedding']
    except Exception:
        return None

def _fallback_embed(t):
    s = sum(bytearray(t.encode('utf-8'))) % 997
    return [float((s + i) % 997) / 997.0 for i in range(128)]

def embed_texts(texts: List[str]):
    if OPENAI_AVAILABLE and openai:
        embs = []
        for t in texts:
            e = get_openai_embedding(t)
            embs.append(e if e else _fallback_embed(t))
        return embs
    if EMBED_MODEL:
        out = EMBED_MODEL.encode(texts)
        return out.tolist() if hasattr(out, "tolist") else out
    return [_fallback_embed(t) for t in texts]

def retrieve_similar_memories(user_id, query, top_k=4):
    mems = DB.list_memories(user_id, limit=200)
    if not mems:
        return []
    texts = [m[1] for m in mems]
    try:
        qv = embed_texts([query])[0]
        cvs = embed_texts(texts)
        if cosine_similarity is None:
            return texts[:top_k]
        import numpy as _np
        sims = cosine_similarity([qv], cvs)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [texts[i] for i in idxs]
    except Exception:
        return texts[:top_k]

# ---- auth (JWT) ----
def create_token(username):
    if not AUTH_AVAILABLE:
        return ""
    payload = {"sub": username, "iat": int(time.time())}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(token) -> Optional[str]:
    if not AUTH_AVAILABLE:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload.get("sub")
    except JWTError:
        return None

def get_current_user(request: Request):
    auth = request.headers.get("authorization")
    if not auth:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = auth.split(" ", 1)[1]
    username = verify_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    row = DB.get_user_by_username(username)
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return {"id": row[0], "username": row[1]}

# ---- models ----
class RegisterReq(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    phone: Optional[str] = None

class LoginReq(BaseModel):
    username_or_email_or_phone: str
    password: str

class ChatReq(BaseModel):
    prompt: str
    mode: Optional[str] = "general"
    file_ids: Optional[List[str]] = []
    language: Optional[str] = None   # ISO language code (e.g. "en", "es", "fr") or None = auto

app = FastAPI(title="MORA — multilingual")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- small helpers ----
def short_read(path, n=3000):
    try:
        return open(path, "rb").read(n).decode("utf-8", errors="ignore")
    except:
        return ""

def summarize_file(path, name):
    try:
        if pd and str(path).lower().endswith((".csv", ".xlsx")):
            df = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
            return f"CSV {name}: {len(df)} rows, columns: {list(df.columns)[:8]}"
        if pymupdf and str(path).lower().endswith(".pdf"):
            doc = pymupdf.open(path); txt = " ".join([p.get_text("text") for p in doc]); return txt[:1500] + ("..." if len(txt) > 1500 else "")
    except Exception:
        pass
    return short_read(path)

def detect_language(text):
    if LANGDETECT_AVAILABLE:
        try:
            return detect(text)
        except:
            return None
    return None

# ---- endpoints: auth, upload, files ----
@app.post("/auth/register")
def register(r: RegisterReq):
    if DB.get_user_by_username(r.username):
        raise HTTPException(400, "Username exists")
    if r.email and DB.get_user_by_email(r.email):
        raise HTTPException(400, "Email exists")
    if r.phone and DB.get_user_by_phone(r.phone):
        raise HTTPException(400, "Phone exists")
    if not AUTH_AVAILABLE:
        raise HTTPException(500, "Auth libs not installed")
    pwd_hash = PWD_CTX.hash(r.password)
    DB.create_user(r.username, r.email, r.phone, pwd_hash)
    token = create_token(r.username)
    return {"ok": True, "access_token": token}

@app.post("/auth/login")
def login(req: LoginReq):
    if not AUTH_AVAILABLE:
        raise HTTPException(500, "Auth libs not installed")
    if "@" in req.username_or_email_or_phone:
        row = DB.get_user_by_email(req.username_or_email_or_phone)
    elif req.username_or_email_or_phone.isdigit():
        row = DB.get_user_by_phone(req.username_or_email_or_phone)
    else:
        row = DB.get_user_by_username(req.username_or_email_or_phone)
    if not row:
        raise HTTPException(401, "User not found")
    if not PWD_CTX.verify(req.password, row[4]):
        raise HTTPException(401, "Bad password")
    token = create_token(row[1])
    return {"access_token": token, "token_type": "bearer"}

@app.post("/upload")
async def upload(file: UploadFile = File(...), user=Depends(get_current_user)):
    fid = str(uuid.uuid4()); dest = UPLOAD_DIR / f"{fid}_{file.filename}"
    data = await file.read(); dest.write_bytes(data)
    DB.add_file(user["id"], fid, file.filename, dest, file.content_type)
    summary = summarize_file(dest, file.filename)
    try:
        e = embed_texts([summary])[0]
        DB.add_memory(user["id"], f"file:{file.filename} summary:{summary}", json.dumps(e))
    except:
        DB.add_memory(user["id"], f"file:{file.filename} summary:{summary}")
    return {"id": fid, "name": file.filename, "summary": summary[:1500]}

@app.get("/files")
def files(user=Depends(get_current_user)):
    return DB.list_files(user["id"])

# ---- multi-backend chooser & wrappers (OpenAI primary) ----
import requests
def call_http_ai(name, prompt):
    key = os.getenv(f"{name.upper()}_API_KEY", "")
    endpoint = os.getenv(f"{name.upper()}_API_ENDPOINT", "")
    if requests and key and endpoint:
        try:
            r = requests.post(endpoint, json={"prompt": prompt, "key": key}, timeout=15)
            if r.ok:
                jd = r.json()
                return jd.get("text") or jd.get("answer") or str(jd)
        except Exception as e:
            return f"{name} error: {e}"
    return f"[{name} stub reply] {prompt}"

def choose_backend(prompt: str, mode: str):
    p = prompt.lower()
    if "image" in p or "diagram" in p:
        return "gemini"
    if mode in ("teacher", "student", "writer"):
        return "claude"
    if "joke" in p or "trending" in p:
        return "grok"
    if "code" in p or mode == "programmer":
        return "openai"
    if mode == "data_analyst" or "analyze" in p:
        return "openai"
    return "openai"

def call_openai_chat(system_prompt, user_prompt, stream=False):
    if not OPENAI_AVAILABLE or openai is None:
        return "OpenAI not configured"
    openai.api_key = OPENAI_KEY
    try:
        if stream:
            resp = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], temperature=0.2, stream=True, max_tokens=1400)
            for chunk in resp:
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    delta = None
                if delta:
                    yield delta
        else:
            resp = openai.ChatCompletion.create(model="gpt-4o", messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], temperature=0.2, max_tokens=900)
            return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI error: {e}"

# ---- chat endpoint with language option ----
@app.post("/chat")
def chat(req: ChatReq, request: Request, user=Depends(get_current_user)):
    # Build context from pinned files & memories & recent chats
    ctx = ""
    for fid in req.file_ids or []:
        r = DB.get_file(fid)
        if not r: continue
        _, uid, name, path, mime = r
        ctx += f"\nFile {name} summary:\n{summarize_file(path, name)}\n"
    memories = retrieve_similar_memories(user["id"], req.prompt, top_k=4)
    if memories:
        ctx += "\nRelevant memories:\n" + "\n".join(memories) + "\n"
    recent = DB.recent_chats(user["id"], limit=8)
    recent_text = "\n".join([f"{r[0]}: {r[1]}" for r in recent])
    system_map = {
        "teacher": "You are an expert teacher. Explain clearly and provide exercises.",
        "student": "You are a friendly tutor who simplifies concepts.",
        "programmer": "You are a senior developer. Provide code, tests, and explanations.",
        "data_analyst": "You are a data analyst. Provide pandas/SQL and visualization guidance.",
        "writer": "You are a professional writer. Help draft and edit."
    }
    system_prompt = system_map.get(req.mode, "You are a helpful assistant.")

    # determine language: user can pass req.language (ISO code), else detect from prompt
    lang = None
    if req.language:
        lang = req.language
    else:
        lang = detect_language(req.prompt) if detect_language(req.prompt) else None

    # Add instruction to respond in target language (if known)
    lang_instruction = f"Reply in {lang}." if lang else ""
    full_prompt = f"{system_prompt}\n{lang_instruction}\nContext:{ctx}\nRecent:{recent_text}\nUser: {req.prompt}"

    DB.add_chat(user["id"], "user", req.prompt)
    backend = choose_backend(req.prompt, req.mode)

    # prefer OpenAI streaming (SSE)
    if backend == "openai" and OPENAI_AVAILABLE and openai is not None:
        def iter_stream():
            gen = call_openai_chat(system_prompt + ("\n" + lang_instruction if lang_instruction else ""), full_prompt, stream=True)
            if isinstance(gen, str):
                DB.add_chat(user["id"], "assistant", gen)
                yield f"data: {json.dumps({'text': gen})}\n\n"
                return
            acc = ""
            for piece in gen:
                acc += piece
                yield f"data: {json.dumps({'text': piece})}\n\n"
            DB.add_chat(user["id"], "assistant", acc)
        return StreamingResponse(iter_stream(), media_type="text/event-stream")

    # fallback single response
    if backend == "openai":
        ans = call_openai_chat(system_prompt + ("\n" + lang_instruction if lang_instruction else ""), full_prompt, stream=False)
    else:
        ans = call_http_ai(backend, full_prompt)

    DB.add_chat(user["id"], "assistant", ans if isinstance(ans, str) else str(ans))
    return JSONResponse({"answer": ans, "language": lang})

# ---- text-to-speech endpoint: returns MP3 ----
@app.post("/speak")
async def speak(text: str = Form(...), language: Optional[str] = Form(None)):
    """
    POST /speak with form fields:
      - text (string)
      - language (optional ISO code, e.g. 'en', 'es', 'fr'). If not provided, auto-detect.
    Returns: audio/mp3 file
    """
    if not text:
        raise HTTPException(400, "Missing text")
    lang = language or (detect_language(text) if detect_language(text) else "en")
    # try gTTS first
    if GTTS_AVAILABLE:
        try:
            tts = gTTS(text, lang=lang)
            fid = f"tts_{uuid.uuid4().hex}.mp3"
            outp = UPLOAD_DIR / fid
            tts.save(str(outp))
            return FileResponse(str(outp), media_type="audio/mpeg", filename=fid)
        except Exception as e:
            # fall through to text response
            pass
    # fallback: return JSON with text & language (client can do TTS)
    return JSONResponse({"text": text, "language": lang, "note": "gTTS not available; client should use native TTS."})

# ---- transcribe endpoint (audio -> text) using OpenAI Whisper if available ----
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), user=Depends(get_current_user)):
    if openai is None or not OPENAI_AVAILABLE:
        raise HTTPException(500, "OpenAI (Whisper) not configured on server")
    # save file temporarily
    tmp = UPLOAD_DIR / f"trans_{uuid.uuid4().hex}_{file.filename}"
    data = await file.read(); tmp.write_bytes(data)
    try:
        openai.api_key = OPENAI_KEY
        # uses OpenAI's speech-to-text endpoint
        audio_file = open(tmp, "rb")
        resp = openai.Audio.transcriptions.create(file=audio_file, model="whisper-1") if hasattr(openai.Audio, "transcriptions") else None
        # note: depending on openai library version you may need openai.Audio.transcribe or openai.Whisper
        if resp is None:
            # try older call
            resp = openai.Transcription.create(file=audio_file, model="whisper-1")
        text = ""
        try:
            text = resp.get("text") if isinstance(resp, dict) else str(resp)
        except:
            text = str(resp)
        # optionally detect language
        lang = detect_language(text) if detect_language(text) else None
        DB.add_memory(user["id"], f"transcription:{text}")
        return {"text": text, "language": lang}
    except Exception as e:
        raise HTTPException(500, f"Whisper error: {e}")
    finally:
        try: tmp.unlink()
        except: pass

@app.get("/health")
def health():
    return {"ok": True, "openai": OPENAI_AVAILABLE, "gtts": GTTS_AVAILABLE, "langdetect": LANGDETECT_AVAILABLE}

# small endpoint to add memory manually
class MemReq(BaseModel):
    text: str
@app.post("/memory/add")
def add_memory(req: MemReq, user=Depends(get_current_user)):
    try:
        e = embed_texts([req.text])[0]
        DB.add_memory(user["id"], req.text, json.dumps(e))
    except Exception:
        DB.add_memory(user["id"], req.text)
    return {"ok": True}
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "🚀 MORA AI Backend is running successfully!"}

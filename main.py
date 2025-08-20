import os
import openai
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# ====== CONFIG ======
# Set your key here or via env var OPENAI_API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_COMPANY_API_KEY")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are CodeGPT, a coding assistant. Be concise and actionable. "
        "Prefer fenced markdown code blocks (```python ... ```). "
        "Preserve formatting and indentation."
    ),
}

# ====== APP ======
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # same-origin by default; keep * for safety during debug
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Single conversation memory
conversation_history = []  # [{role, content}, ...]
MAX_TURNS = 40

def _trim(msgs, limit=MAX_TURNS):
    return msgs[-limit * 2 :] if len(msgs) > limit * 2 else msgs

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    """
    Body: { "message": str }
    Returns: { "ok": bool, "reply": str, "error": str|null }
    """
    global conversation_history
    try:
        data = await request.json()
    except Exception as e:
        print("[/chat] JSON parse error:", e)
        return JSONResponse({"ok": False, "reply": "", "error": f"Bad JSON: {e}"}, status_code=400)

    user_text = data.get("message", "")
    if not isinstance(user_text, str) or user_text == "":
        return JSONResponse({"ok": False, "reply": "", "error": "Empty input."}, status_code=400)

    # Build messages
    conversation_history.append({"role": "user", "content": user_text})
    conversation_history[:] = _trim(conversation_history)
    messages = [SYSTEM_PROMPT] + conversation_history

    print(f"[/chat] Calling OpenAI model={DEFAULT_MODEL}, tokens≈, last_user_len={len(user_text)}")
    try:
        resp = openai.ChatCompletion.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.15,
            max_tokens=900,
        )
        reply = resp["choices"][0]["message"]["content"]
        conversation_history.append({"role": "assistant", "content": reply})
        conversation_history[:] = _trim(conversation_history)
        print("[/chat] Success, reply_len=", len(reply))
        return JSONResponse({"ok": True, "reply": reply, "error": None})
    except Exception as e:
        # Surface the exact error back to UI so you can see it
        err = f"{type(e).__name__}: {e}"
        print("[/chat] OpenAI ERROR:", err)
        return JSONResponse({"ok": False, "reply": "", "error": err}, status_code=200)

@app.post("/reset")
async def reset():
    global conversation_history
    conversation_history = []
    print("[/reset] Conversation cleared")
    return JSONResponse({"ok": True, "status": "reset"})

@app.get("/health")
async def health():
    return {"ok": True}

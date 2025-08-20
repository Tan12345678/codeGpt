import openai
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ====== CONFIG ======
openai.api_key = "YOUR_COMPANY_API_KEY"   # <-- put your key
DEFAULT_MODEL = "gpt-4o-mini"             # or your allowed model (e.g., "gpt-4o")

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
templates = Jinja2Templates(directory="templates")

# In-memory single-thread conversation (no sidebar, one chat at a time)
conversation_history = []  # [{role, content}, ...]

MAX_TURNS = 40  # bounds memory (user+assistant pairs)

def _trim(msgs, limit=MAX_TURNS):
    if len(msgs) > limit * 2:
        return msgs[-limit * 2 :]
    return msgs

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    """Body: { "message": str }"""
    global conversation_history
    data = await request.json()
    user_text = data.get("message", "")

    if not isinstance(user_text, str) or user_text == "":
        return JSONResponse({"reply": "⚠️ Empty input."})

    # append user → build messages with system prompt
    conversation_history.append({"role": "user", "content": user_text})
    conversation_history[:] = _trim(conversation_history)
    messages = [SYSTEM_PROMPT] + conversation_history

    try:
        resp = openai.ChatCompletion.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.15,
            max_tokens=900,
        )
        reply = resp["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"⚠️ Error: {e}"

    # save assistant
    conversation_history.append({"role": "assistant", "content": reply})
    conversation_history[:] = _trim(conversation_history)

    return JSONResponse({"reply": reply})

@app.post("/reset")
async def reset():
    global conversation_history
    conversation_history = []
    return JSONResponse({"status": "reset"})

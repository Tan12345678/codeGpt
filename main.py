import uuid
import time
import openai
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ========== CONFIG ==========
openai.api_key = "YOUR_COMPANY_API_KEY"  # <- put your key here
DEFAULT_MODEL = "gpt-4o-mini"            # change if necessary
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are CodeGPT, a coding assistant. "
        "Be concise and actionable. Prefer fenced markdown code blocks "
        "for code (```python ... ```). Preserve formatting and indentation."
    )
}
MAX_TURNS = 40  # keep memory bounded

# ========== APP ==========
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# In-memory store for chats:
# chats[chat_id] = { title, created, updated, messages: [{role,content}, ...] }
chats: Dict[str, Dict[str, Any]] = {}


def _now_ts() -> int:
    return int(time.time())


def _new_chat(title: str = "New Chat") -> str:
    cid = str(uuid.uuid4())
    now = _now_ts()
    chats[cid] = {"title": title, "created": now, "updated": now, "messages": []}
    return cid


def _trim_messages(msgs: List[Dict[str, str]], limit: int = MAX_TURNS) -> List[Dict[str, str]]:
    if len(msgs) > limit * 2:
        return msgs[-limit * 2 :]
    return msgs


def _compact_title(text: str) -> str:
    t = " ".join(text.strip().split())
    return (t[:48] + "…") if len(t) > 48 else (t or "New Chat")


# ========== ROUTES ==========
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/chats")
async def list_chats():
    items = [
        {
            "id": cid,
            "title": data["title"],
            "created": data["created"],
            "updated": data["updated"],
        }
        for cid, data in chats.items()
    ]
    items.sort(key=lambda x: x["updated"], reverse=True)
    return JSONResponse({"chats": items})


@app.post("/new_chat")
async def new_chat():
    cid = _new_chat()
    return JSONResponse({"chat_id": cid})


@app.get("/messages")
async def get_messages(chat_id: str):
    if chat_id not in chats:
        return JSONResponse({"messages": []})
    return JSONResponse({"messages": chats[chat_id]["messages"], "title": chats[chat_id]["title"]})


@app.post("/chat")
async def chat(request: Request):
    """
    Body:
      { "chat_id": str (optional), "message": str }
    """
    data = await request.json()
    user_text: str = data.get("message", "")
    chat_id: Optional[str] = data.get("chat_id")

    if not isinstance(user_text, str) or user_text == "":
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # create chat if needed
    if not chat_id or chat_id not in chats:
        chat_id = _new_chat()

    # set title from first user message
    if not chats[chat_id]["messages"]:
        chats[chat_id]["title"] = _compact_title(user_text)

    # append user message (preserve exact formatting)
    chats[chat_id]["messages"].append({"role": "user", "content": user_text})
    chats[chat_id]["messages"] = _trim_messages(chats[chat_id]["messages"])

    # create payload for OpenAI
    messages = [SYSTEM_PROMPT] + chats[chat_id]["messages"]

    try:
        resp = openai.ChatCompletion.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0.15,
            max_tokens=900,
        )
        reply = resp["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"Error: {e}"

    # save assistant reply
    chats[chat_id]["messages"].append({"role": "assistant", "content": reply})
    chats[chat_id]["messages"] = _trim_messages(chats[chat_id]["messages"])
    chats[chat_id]["updated"] = _now_ts()

    return JSONResponse({"chat_id": chat_id, "reply": reply})


@app.post("/reset_chat")
async def reset_chat(request: Request):
    data = await request.json()
    chat_id = data.get("chat_id")
    if chat_id and chat_id in chats:
        chats[chat_id]["messages"] = []
        chats[chat_id]["updated"] = _now_ts()
        return JSONResponse({"status": "ok"})
    return JSONResponse({"status": "no-chat"}, status_code=400)


@app.post("/delete_chat")
async def delete_chat(request: Request):
    data = await request.json()
    chat_id = data.get("chat_id")
    if chat_id and chat_id in chats:
        del chats[chat_id]
        return JSONResponse({"status": "deleted"})
    return JSONResponse({"status": "no-chat"}, status_code=400)

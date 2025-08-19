import openai
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ✅ Company API key
openai.api_key = "YOUR_COMPANY_API_KEY"

app = FastAPI()

# Memory for conversation
conversation_history = []

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are CodeGPT, a coding assistant. "
        "Always return code inside proper markdown ```language blocks``` "
        "so the UI can format it nicely."
    )
}

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    global conversation_history
    data = await request.json()
    user_message = data["message"]

    # Append user message to history
    conversation_history.append({"role": "user", "content": user_message})

    # Create messages with system prompt
    messages = [SYSTEM_PROMPT] + conversation_history

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or your local model
        messages=messages,
        max_tokens=800,
        temperature=0.2
    )

    reply = response["choices"][0]["message"]["content"]

    # Save reply to memory
    conversation_history.append({"role": "assistant", "content": reply})

    return JSONResponse({"reply": reply})

@app.post("/reset")
async def reset():
    global conversation_history
    conversation_history = []
    return JSONResponse({"status": "reset"})

# main.py
import os
import base64
import uuid
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY="Enter_Your_API_Here"
BASE_URL = "https://api.groq.com/openai/v1"  # Groq OpenAI-compatible base
DEFAULT_MODEL = os.getenv("DEFAULT_GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    # We allow running without key for local dev; endpoints will return helpful error.
    print("Warning: GROQ_API_KEY not set. LLM calls will fail until you set it in .env")

app = FastAPI(title="Groq LLM FastAPI Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# In-memory CRUD store
# -------------------------
class Item(BaseModel):
    id: Optional[str]
    title: str
    content: str

_items_store = {}  # id -> Item dict

@app.get("/items", response_model=List[Item])
async def list_items():
    return list(_items_store.values())

@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: str):
    item = _items_store.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.post("/items", response_model=Item, status_code=201)
async def create_item(payload: Item):
    item_id = payload.id or str(uuid.uuid4())
    payload.id = item_id
    _items_store[item_id] = payload
    return payload

@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: str, payload: Item):
    if item_id not in _items_store:
        raise HTTPException(status_code=404, detail="Item not found")
    payload.id = item_id
    _items_store[item_id] = payload
    return payload

@app.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: str):
    if item_id in _items_store:
        del _items_store[item_id]
    return {"detail": "deleted"}

# -------------------------
# Helpers for Groq calls
# -------------------------
async def groq_responses_request(body: dict, timeout: int = 60):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set. Put it in .env or environment")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL}/responses"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=body, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # return detailed error for debugging
            raise HTTPException(status_code=500, detail={"status_code": r.status_code, "text": r.text})
        return r.json()

# -------------------------
# LLM endpoints
# -------------------------
class TextRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0

@app.post("/llm/text")
async def llm_text(req: TextRequest):
    model = req.model or DEFAULT_MODEL
    body = {
        "model": model,
        "input": req.prompt,
        "temperature": req.temperature,
        "max_output_tokens": req.max_tokens,
    }
    result = await groq_responses_request(body)
    # result schema varies; try to extract highest-value text fields
    # Common field: result["output"][0]["content"][0]["text"] or result["output_text"]
    output_text = result.get("output_text")
    if not output_text:
        # try nested locations
        outputs = result.get("output") or result.get("outputs") or []
        if isinstance(outputs, list) and outputs:
            # attempt to join textual parts
            parts = []
            for o in outputs:
                # some responses contain "content" list with dicts that have "text"
                cont = o.get("content") or []
                for c in cont:
                    text = c.get("text") if isinstance(c, dict) else None
                    if text:
                        parts.append(text)
            output_text = "\n".join(parts) if parts else str(outputs)
    return {"model": model, "result_raw": result, "output_text": output_text}

class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.0

@app.post("/llm/chat")
async def llm_chat(req: ChatRequest):
    model = req.model or DEFAULT_MODEL
    # Groq supports OpenAI-compatible chat endpoint and Responses endpoint.
    # We'll use Responses endpoint with `input` as conversation; many Groq examples allow passing a list of messages.
    # For safety we pass a concatenated string of the conversation if the API returns error with nested format.
    messages = req.messages
    # Try to build a messages list in OpenAI chat format:
    body = {
        "model": model,
        "input": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": req.temperature,
        "max_output_tokens": req.max_tokens,
    }
    result = await groq_responses_request(body)
    return {"model": model, "result_raw": result}

# -------------------------
# Image endpoint: accepts uploaded image and text instruction
# -------------------------
# @app.post("/llm/image")
# async def llm_image(file: UploadFile = File(...), instruction: str = Form(...), model: Optional[str] = Form(None)):
#     """
#     Receives an uploaded image + instruction text, encodes the image to base64,
#     and sends it to Groq Responses API as part of the input.
#     Note: Groq Responses API accepts images either as URLs or base64 data URLs.
#     See the Groq docs for exact payload shapes if API versions change. :contentReference[oaicite:3]{index=3}
#     """
#     model = model or DEFAULT_MODEL
#     contents = await file.read()
#     b64 = base64.b64encode(contents).decode("utf-8")
#     data_url = f"data:{file.content_type};base64,{b64}"

#     # Build a simple payload that includes the image as a data URL + instruction text.
#     # If Groq changes the shape, adjust per official docs. (We return raw result so you can inspect.)
#     body = {
#         "model": model,
#         #"input": [
#         "massage": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "input_text", "text": instruction},
#                     {"type": "input_image", "image_url": data_url}
#                 ]
#             }
#         ],
#         "temperature": 0.0,
#         "max_output_tokens": 512,
#     }

#     result = await groq_responses_request(body, timeout=120)
#     return {"model": model, "result_raw": result}
@app.post("/llm/image")
async def llm_image(
    file: UploadFile = File(...),
    instruction: str = Form(...),
    model: Optional[str] = Form(None)
):
    """
    Receives an uploaded image + instruction text, encodes it to base64,
    and sends it to Groq Chat Completions API as a multimodal input.
    """
    model = model or DEFAULT_MODEL
    contents = await file.read()
    b64 = base64.b64encode(contents).decode("utf-8")
    data_url = f"data:{file.content_type};base64,{b64}"

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": data_url},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    result = await groq_chat_request(body, timeout=120)

    # Extract text output
    output_text = None
    try:
        output_text = result["choices"][0]["message"]["content"]
    except Exception:
        output_text = str(result)

    return {"model": model, "result_raw": result, "output_text": output_text}


@app.get("/llm/models")
async def suggested_models():
    # Minimal helpful list; update to whatever models your Groq account has access to.
    return {
        "default": DEFAULT_MODEL,
        "suggested_free_preview": [
            "openai/gpt-oss-20b",
            "mixtral-8x7b",
            "openai/llama3-7b"
        ],
        "note": "Check your Groq console for available models and permissions."
    }

# run: uvicorn main:app --host 0.0.0.0 --port 8000

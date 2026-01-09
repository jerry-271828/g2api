import os
import json
import uuid
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from gumloop_types import (
    ClaudeRequest, ChatCompletionRequest, ResponsesRequest, GeminiRequest
)
from gumloop_client import send_chat, GumloopStreamHandler
from gumloop_parser import (
    build_message_start, build_content_block_start, build_content_block_delta,
    build_content_block_stop, build_ping, build_message_delta, build_message_stop,
    build_openai_chunk, build_openai_done, build_gemini_response, build_gemini_stream_chunk,
    build_tool_use_start, build_tool_use_delta
)
from tool_converter import (
    convert_messages_with_tools, parse_tool_calls, detect_tool_loop
)
from auth import get_auth
import db

load_dotenv()

app = FastAPI(title="Gumloop 2API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GUMMIE_ID = os.getenv("GUMLOOP_GUMMIE_ID", "")
ALLOWED_KEYS = [k.strip() for k in os.getenv("OPENAI_KEYS", "").split(",") if k.strip()]
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

FRONTEND_DIR = Path(__file__).parent / "frontend"

# Pydantic models for admin API
class LoginRequest(BaseModel):
    password: str

class AccountCreate(BaseModel):
    label: Optional[str] = None
    api_key: Optional[str] = None
    gummie_id: Optional[str] = None
    other: Optional[Dict] = None
    enabled: bool = True

class AccountUpdate(BaseModel):
    label: Optional[str] = None
    api_key: Optional[str] = None
    gummie_id: Optional[str] = None
    other: Optional[Dict] = None
    enabled: Optional[bool] = None

class GumloopLoginRequest(BaseModel):
    email: str
    password: str
    label: Optional[str] = None
    gummie_id: Optional[str] = None
    enabled: bool = True

# Admin authentication
def verify_admin(authorization: Optional[str] = Header(None)) -> bool:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized")
    if authorization[7:] != ADMIN_PASSWORD:
        raise HTTPException(401, "Invalid password")
    return True

# Startup/shutdown events
@app.on_event("startup")
async def startup():
    await db.init_db()

@app.on_event("shutdown")
async def shutdown():
    await db.close_db()

def _extract_key(auth: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    if auth and auth.startswith("Bearer "):
        return auth[7:].strip()
    return x_api_key

async def verify_key(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    key = _extract_key(authorization, x_api_key)
    if ALLOWED_KEYS and (not key or key not in ALLOWED_KEYS):
        raise HTTPException(401, "Invalid API key")
    return key

def map_model(model: str) -> str:
    m = model.lower()
    if "opus" in m or "gpt-4-turbo" in m:
        return "claude-opus-4-5"
    if "haiku" in m or "gpt-3.5" in m:
        return "claude-haiku-4-5"
    return "claude-sonnet-4-5"

def convert_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    result = []
    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if isinstance(content, list):
            text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            content = "\n".join(text_parts)
        result.append({"role": role, "content": str(content)})
    return result

def validate_messages(messages: List[Dict[str, Any]], max_messages: int = 100) -> None:
    """Validate messages to prevent potential issues."""
    if len(messages) > max_messages:
        raise HTTPException(400, f"Too many messages: {len(messages)} > {max_messages}")

    if not messages:
        raise HTTPException(400, "No messages provided")

    # Check last message is from user
    if messages[-1].get("role") != "user":
        raise HTTPException(400, "Last message must be from user")

    # Validate message alternation (allow some flexibility for tool_result messages)
    prev_role = None
    consecutive_same_role = 0
    for msg in messages:
        role = msg.get("role", "user")
        if role == prev_role:
            consecutive_same_role += 1
            if consecutive_same_role >= 3:
                raise HTTPException(400, f"Too many consecutive {role} messages (possible loop)")
        else:
            consecutive_same_role = 0
        prev_role = role

# Claude Messages API
@app.post("/v1/messages")
async def claude_messages(req: ClaudeRequest, _: str = Depends(verify_key)):
    gummie_id = GUMMIE_ID
    if not gummie_id:
        raise HTTPException(500, "GUMLOOP_GUMMIE_ID not configured")

    # Check for tool call loops
    if req.tools:
        loop_error = detect_tool_loop([{"role": m.role, "content": m.content} for m in req.messages])
        if loop_error:
            raise HTTPException(400, loop_error)

    # Convert messages with tool support
    sys_text = None
    if req.system:
        sys_text = req.system if isinstance(req.system, str) else "\n".join(
            b.get("text", "") for b in req.system if isinstance(b, dict) and b.get("type") == "text"
        )

    tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in req.tools] if req.tools else None
    messages = convert_messages_with_tools(
        [{"role": m.role, "content": m.content} for m in req.messages],
        tools=tools,
        system=sys_text
    )

    validate_messages(messages)
    model = map_model(req.model)
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    thinking_enabled = bool(req.thinking and req.thinking.get("type") == "enabled")
    has_tools = bool(req.tools)

    if req.stream:
        async def stream_gen():
            handler = GumloopStreamHandler(model=model)
            yield build_message_start(msg_id, model, 0)
            yield build_ping()
            block_idx = 0
            in_thinking = False
            in_text = False
            full_text = ""

            async for event in send_chat(gummie_id, messages):
                ev = handler.handle_event(event)
                ev_type = ev.get("type")

                if ev_type == "reasoning_start" and thinking_enabled:
                    yield build_content_block_start(block_idx, "thinking")
                    in_thinking = True
                elif ev_type == "reasoning_delta" and ev.get("delta") and thinking_enabled:
                    if not in_thinking:
                        yield build_content_block_start(block_idx, "thinking")
                        in_thinking = True
                    yield build_content_block_delta(block_idx, ev["delta"], "thinking_delta", "thinking")
                elif ev_type == "reasoning_end" and thinking_enabled:
                    if in_thinking:
                        yield build_content_block_stop(block_idx)
                        block_idx += 1
                        in_thinking = False
                elif ev_type == "text_start":
                    yield build_content_block_start(block_idx, "text")
                    in_text = True
                elif ev_type == "text_delta" and ev.get("delta"):
                    if not in_text:
                        yield build_content_block_start(block_idx, "text")
                        in_text = True
                    full_text += ev["delta"]
                    yield build_content_block_delta(block_idx, ev["delta"])
                elif ev_type == "text_end":
                    if in_text:
                        yield build_content_block_stop(block_idx)
                        block_idx += 1
                        in_text = False
                elif ev_type == "finish":
                    if in_thinking:
                        yield build_content_block_stop(block_idx)
                        block_idx += 1
                    if in_text:
                        yield build_content_block_stop(block_idx)
                        block_idx += 1

                    # Parse tool calls from response if tools are enabled
                    stop_reason = "end_turn"
                    if has_tools:
                        remaining_text, tool_uses = parse_tool_calls(full_text)
                        if tool_uses:
                            stop_reason = "tool_use"
                            for tu in tool_uses:
                                yield build_tool_use_start(block_idx, tu["id"], tu["name"])
                                yield build_tool_use_delta(block_idx, json.dumps(tu["input"], ensure_ascii=False))
                                yield build_content_block_stop(block_idx)
                                block_idx += 1

                    yield build_message_delta(ev["usage"]["output_tokens"], stop_reason)
                    yield build_message_stop()
                    break

        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        handler = GumloopStreamHandler(model=model)
        async for event in send_chat(gummie_id, messages):
            handler.handle_event(event)

        full_text = handler.get_full_text()
        content = []
        stop_reason = "end_turn"

        if thinking_enabled and handler.get_full_reasoning():
            content.append({"type": "thinking", "thinking": handler.get_full_reasoning()})

        # Parse tool calls from response
        if has_tools:
            remaining_text, tool_uses = parse_tool_calls(full_text)
            if remaining_text:
                content.append({"type": "text", "text": remaining_text})
            if tool_uses:
                stop_reason = "tool_use"
                for tu in tool_uses:
                    content.append(tu)
        else:
            content.append({"type": "text", "text": full_text})

        return JSONResponse({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": handler.input_tokens, "output_tokens": handler.output_tokens}
        })

# OpenAI Chat Completions API
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _: str = Depends(verify_key)):
    gummie_id = GUMMIE_ID
    if not gummie_id:
        raise HTTPException(500, "GUMLOOP_GUMMIE_ID not configured")

    messages = convert_messages(req.messages)
    validate_messages(messages)
    model = map_model(req.model or "gpt-4")
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if req.stream:
        async def stream_gen():
            handler = GumloopStreamHandler(model=model)
            yield build_openai_chunk(stream_id, model, role="assistant", created=created)

            async for event in send_chat(gummie_id, messages):
                ev = handler.handle_event(event)
                if ev.get("type") == "text_delta" and ev.get("delta"):
                    yield build_openai_chunk(stream_id, model, content=ev["delta"], created=created)
                elif ev.get("type") == "finish":
                    yield build_openai_chunk(stream_id, model, finish_reason="stop", created=created)
                    yield build_openai_done()
                    break

        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        handler = GumloopStreamHandler(model=model)
        async for event in send_chat(gummie_id, messages):
            handler.handle_event(event)

        return JSONResponse({
            "id": stream_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": handler.get_full_text()}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": handler.input_tokens, "completion_tokens": handler.output_tokens, "total_tokens": handler.input_tokens + handler.output_tokens}
        })

# OpenAI Responses API
@app.post("/v1/responses")
async def responses(req: ResponsesRequest, _: str = Depends(verify_key)):
    gummie_id = GUMMIE_ID
    if not gummie_id:
        raise HTTPException(500, "GUMLOOP_GUMMIE_ID not configured")

    input_text = req.input if isinstance(req.input, str) else "\n".join(
        p.get("text", "") for p in req.input if isinstance(p, dict) and p.get("type") == "text"
    )
    messages = [{"role": "user", "content": input_text}]
    if req.instructions:
        messages.insert(0, {"role": "user", "content": f"[Instructions]: {req.instructions}"})

    model = map_model(req.model or "gpt-4")
    resp_id = f"resp_{uuid.uuid4().hex}"

    if req.stream:
        async def stream_gen():
            handler = GumloopStreamHandler(model=model)
            async for event in send_chat(gummie_id, messages):
                ev = handler.handle_event(event)
                if ev.get("type") == "text_delta" and ev.get("delta"):
                    yield f"data: {json.dumps({'type': 'content_part_delta', 'delta': {'text': ev['delta']}})}\n\n"
                elif ev.get("type") == "finish":
                    yield f"data: {json.dumps({'type': 'response_done', 'response': {'id': resp_id, 'output': [{'type': 'text', 'text': handler.get_full_text()}]}})}\n\n"
                    break

        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        handler = GumloopStreamHandler(model=model)
        async for event in send_chat(gummie_id, messages):
            handler.handle_event(event)

        return JSONResponse({
            "id": resp_id,
            "object": "response",
            "model": model,
            "output": [{"type": "text", "text": handler.get_full_text()}],
            "usage": {"input_tokens": handler.input_tokens, "output_tokens": handler.output_tokens}
        })

# Gemini API
@app.post("/v1beta/models/{model}:generateContent")
async def gemini_generate(model: str, req: GeminiRequest, _: str = Depends(verify_key)):
    gummie_id = GUMMIE_ID
    if not gummie_id:
        raise HTTPException(500, "GUMLOOP_GUMMIE_ID not configured")

    messages = []
    for content in req.contents:
        role = content.role or "user"
        text = "\n".join(p.text or "" for p in content.parts if p.text)
        messages.append({"role": role, "content": text})

    mapped_model = map_model(model)
    handler = GumloopStreamHandler(model=mapped_model)
    async for event in send_chat(gummie_id, messages):
        handler.handle_event(event)

    return JSONResponse(build_gemini_response(handler.get_full_text(), mapped_model))

@app.post("/v1beta/models/{model}:streamGenerateContent")
async def gemini_stream(model: str, req: GeminiRequest, _: str = Depends(verify_key)):
    gummie_id = GUMMIE_ID
    if not gummie_id:
        raise HTTPException(500, "GUMLOOP_GUMMIE_ID not configured")

    messages = []
    for content in req.contents:
        role = content.role or "user"
        text = "\n".join(p.text or "" for p in content.parts if p.text)
        messages.append({"role": role, "content": text})

    mapped_model = map_model(model)

    async def stream_gen():
        handler = GumloopStreamHandler(model=mapped_model)
        async for event in send_chat(gummie_id, messages):
            ev = handler.handle_event(event)
            if ev.get("type") == "text_delta" and ev.get("delta"):
                yield build_gemini_stream_chunk(ev["delta"])

    return StreamingResponse(stream_gen(), media_type="application/x-ndjson")

@app.get("/healthz")
async def health():
    return {"status": "ok"}

@app.get("/v1/models")
async def list_models():
    return {"data": [
        {"id": "claude-sonnet-4-5", "object": "model"},
        {"id": "claude-opus-4-5", "object": "model"},
        {"id": "claude-haiku-4-5", "object": "model"},
    ]}

# ============ Admin API ============

@app.post("/api/login")
async def api_login(req: LoginRequest):
    if req.password == ADMIN_PASSWORD:
        return {"success": True}
    return {"success": False, "message": "Invalid password"}

@app.get("/v2/accounts")
async def list_accounts(
    enabled: Optional[bool] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    _: bool = Depends(verify_admin)
):
    accounts = await db.get_accounts(enabled=enabled, sort_by=sort_by, sort_order=sort_order)
    return {"accounts": accounts, "count": len(accounts)}

@app.post("/v2/accounts")
async def create_account(req: AccountCreate, _: bool = Depends(verify_admin)):
    account = await db.create_account(req.dict())
    return account

@app.get("/v2/accounts/{account_id}")
async def get_account(account_id: str, _: bool = Depends(verify_admin)):
    account = await db.get_account(account_id)
    if not account:
        raise HTTPException(404, "Account not found")
    return account

@app.patch("/v2/accounts/{account_id}")
async def update_account(account_id: str, req: AccountUpdate, _: bool = Depends(verify_admin)):
    data = {k: v for k, v in req.dict().items() if v is not None}
    account = await db.update_account(account_id, data)
    if not account:
        raise HTTPException(404, "Account not found")
    return account

@app.delete("/v2/accounts/{account_id}")
async def delete_account(account_id: str, _: bool = Depends(verify_admin)):
    success = await db.delete_account(account_id)
    if not success:
        raise HTTPException(404, "Account not found")
    return {"success": True}

@app.post("/v2/auth/gumloop")
async def gumloop_login(req: GumloopLoginRequest, _: bool = Depends(verify_admin)):
    """Login to Gumloop with email/password and create account"""
    from auth import GumloopAuth, _get_http_client

    auth = GumloopAuth(req.email, req.password)
    try:
        async with _get_http_client() as client:
            data = await auth.login(client)
    except Exception as e:
        raise HTTPException(400, f"Gumloop login failed: {str(e)}")

    account = await db.create_account({
        "label": req.label or req.email,
        "api_key": data.get("idToken"),
        "gummie_id": req.gummie_id or GUMMIE_ID,
        "enabled": req.enabled,
        "other": {
            "email": req.email,
            "refresh_token": data.get("refreshToken"),
            "user_id": data.get("localId")
        }
    })
    return {"success": True, "account": account}

# ============ Frontend Pages ============

@app.get("/login")
async def login_page():
    return FileResponse(FRONTEND_DIR / "login.html")

@app.get("/")
async def index_page():
    return FileResponse(FRONTEND_DIR / "index.html")

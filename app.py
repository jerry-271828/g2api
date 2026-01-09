import os
import json
import uuid
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

from gumloop_types import (
    ClaudeRequest, ChatCompletionRequest, ResponsesRequest, GeminiRequest
)
from gumloop_client import send_chat, GumloopStreamHandler
from gumloop_parser import (
    build_message_start, build_content_block_start, build_content_block_delta,
    build_content_block_stop, build_ping, build_message_delta, build_message_stop,
    build_openai_chunk, build_openai_done, build_gemini_response, build_gemini_stream_chunk
)
from auth import get_auth

load_dotenv()

app = FastAPI(title="Gumloop 2API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GUMMIE_ID = os.getenv("GUMLOOP_GUMMIE_ID", "")
ALLOWED_KEYS = [k.strip() for k in os.getenv("OPENAI_KEYS", "").split(",") if k.strip()]

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

# Claude Messages API
@app.post("/v1/messages")
async def claude_messages(req: ClaudeRequest, _: str = Depends(verify_key)):
    gummie_id = GUMMIE_ID
    if not gummie_id:
        raise HTTPException(500, "GUMLOOP_GUMMIE_ID not configured")

    messages = convert_messages(req.messages)
    if req.system:
        sys_text = req.system if isinstance(req.system, str) else "\n".join(
            b.get("text", "") for b in req.system if isinstance(b, dict) and b.get("type") == "text"
        )
        if sys_text:
            messages.insert(0, {"role": "user", "content": f"[System]: {sys_text}"})

    model = map_model(req.model)
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Check if thinking is enabled
    thinking_enabled = bool(req.thinking and req.thinking.get("type") == "enabled")

    if req.stream:
        async def stream_gen():
            handler = GumloopStreamHandler(model=model)
            yield build_message_start(msg_id, model, 0)
            yield build_ping()
            block_idx = 0
            in_thinking = False
            in_text = False

            async for event in send_chat(gummie_id, messages):
                ev = handler.handle_event(event)
                ev_type = ev.get("type")

                # Handle thinking/reasoning blocks
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

                # Handle text blocks
                elif ev_type == "text_start":
                    yield build_content_block_start(block_idx, "text")
                    in_text = True
                elif ev_type == "text_delta" and ev.get("delta"):
                    if not in_text:
                        yield build_content_block_start(block_idx, "text")
                        in_text = True
                    yield build_content_block_delta(block_idx, ev["delta"])
                elif ev_type == "text_end":
                    if in_text:
                        yield build_content_block_stop(block_idx)
                        block_idx += 1
                        in_text = False
                elif ev_type == "finish":
                    if in_thinking:
                        yield build_content_block_stop(block_idx)
                    if in_text:
                        yield build_content_block_stop(block_idx)
                    yield build_message_delta(ev["usage"]["output_tokens"])
                    yield build_message_stop()
                    break

        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        handler = GumloopStreamHandler(model=model)
        async for event in send_chat(gummie_id, messages):
            handler.handle_event(event)

        content = []
        if thinking_enabled and handler.get_full_reasoning():
            content.append({"type": "thinking", "thinking": handler.get_full_reasoning()})
        content.append({"type": "text", "text": handler.get_full_text()})

        return JSONResponse({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": handler.input_tokens, "output_tokens": handler.output_tokens}
        })

# OpenAI Chat Completions API
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _: str = Depends(verify_key)):
    gummie_id = GUMMIE_ID
    if not gummie_id:
        raise HTTPException(500, "GUMLOOP_GUMMIE_ID not configured")

    messages = convert_messages(req.messages)
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

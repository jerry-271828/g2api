import json
from typing import Optional, Dict, Any

def _sse_format(event_type: str, data: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

# Claude SSE builders
def build_message_start(msg_id: str, model: str = "claude-sonnet-4-5", input_tokens: int = 0) -> str:
    return _sse_format("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}
        }
    })

def build_content_block_start(index: int, block_type: str = "text") -> str:
    block = {"type": block_type, "text": ""} if block_type == "text" else {"type": block_type, "thinking": ""}
    return _sse_format("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})

def build_content_block_delta(index: int, text: str, delta_type: str = "text_delta", field: str = "text") -> str:
    delta = {"type": delta_type, field: text}
    return _sse_format("content_block_delta", {"type": "content_block_delta", "index": index, "delta": delta})

def build_content_block_stop(index: int) -> str:
    return _sse_format("content_block_stop", {"type": "content_block_stop", "index": index})

def build_tool_use_start(index: int, tool_id: str, name: str) -> str:
    return _sse_format("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}}
    })

def build_tool_use_delta(index: int, input_json: str) -> str:
    return _sse_format("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": input_json}
    })

def build_ping() -> str:
    return _sse_format("ping", {"type": "ping"})

def build_message_delta(output_tokens: int, stop_reason: str = "end_turn") -> str:
    return _sse_format("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens}
    })

def build_message_stop() -> str:
    return _sse_format("message_stop", {"type": "message_stop"})

# OpenAI SSE builders
def build_openai_chunk(stream_id: str, model: str, content: Optional[str] = None, role: Optional[str] = None, finish_reason: Optional[str] = None, created: int = 0) -> str:
    delta = {}
    if role:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    chunk = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

def build_openai_done() -> str:
    return "data: [DONE]\n\n"

# Gemini response builders
def build_gemini_response(text: str, model: str, finish_reason: str = "STOP") -> Dict[str, Any]:
    return {
        "candidates": [{
            "content": {"parts": [{"text": text}], "role": "model"},
            "finishReason": finish_reason,
            "index": 0
        }],
        "modelVersion": model
    }

def build_gemini_stream_chunk(text: str) -> str:
    return json.dumps({"candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "index": 0}]}) + "\n"

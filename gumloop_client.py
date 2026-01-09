import os
import json
import uuid
import asyncio
from typing import Optional, Dict, Any, List, AsyncGenerator
import websockets
from auth import get_auth

WS_URL = "wss://ws.gumloop.com/ws/gummies"

def _get_proxy() -> Optional[str]:
    """Get proxy URL from environment."""
    return os.getenv("HTTP_PROXY", "").strip() or None

async def send_chat(
    gummie_id: str,
    messages: List[Dict[str, Any]],
    interaction_id: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Send chat message via WebSocket and yield response events."""
    auth = get_auth()
    id_token = await auth.get_token()

    if not interaction_id:
        interaction_id = str(uuid.uuid4()).replace("-", "")[:22]

    # Build Gumloop message format
    gumloop_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_id = msg.get("id", f"msg_{uuid.uuid4().hex[:24]}")

        if role == "assistant":
            gumloop_msgs.append({
                "id": msg_id,
                "role": "assistant",
                "parts": [{"id": f"{msg_id}_part", "type": "text", "text": content}]
            })
        else:
            gumloop_msgs.append({
                "id": msg_id,
                "role": "user",
                "content": content
            })

    payload = {
        "type": "start",
        "payload": {
            "id_token": id_token,
            "context": {
                "chat": {"id": interaction_id, "msgs": gumloop_msgs},
                "type": "chat",
                "gummie_id": gummie_id
            }
        }
    }

    # WebSocket connection with optional proxy
    proxy = _get_proxy()
    ws_kwargs = {"additional_headers": {"Origin": "https://www.gumloop.com"}}

    if proxy:
        # websockets library supports proxy via environment or explicit parameter
        # For SOCKS proxy: pip install websockets[socks]
        # For HTTP proxy: use websockets-proxy or set environment
        import os
        os.environ.setdefault("https_proxy", proxy)
        os.environ.setdefault("http_proxy", proxy)

    async with websockets.connect(WS_URL, **ws_kwargs) as ws:
        await ws.send(json.dumps(payload))

        async for message in ws:
            try:
                event = json.loads(message)
                yield event
                if event.get("type") == "finish":
                    break
            except json.JSONDecodeError:
                continue

class GumloopStreamHandler:
    """Handle Gumloop WebSocket events and convert to target format."""

    def __init__(self, model: str = "claude-sonnet-4-5", input_tokens: int = 0):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = 0
        self.text_buffer: List[str] = []
        self.reasoning_buffer: List[str] = []
        self.block_index = -1
        self.in_text = False
        self.in_reasoning = False
        self.message_started = False
        self.finished = False
        self.response_ended = False  # Prevent processing after response ends

    def handle_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single event and return normalized data."""
        # Early return if response has already ended
        if self.response_ended:
            return {"type": "ignored", "reason": "response_ended"}

        event_type = event.get("type", "")

        if event_type == "step-start":
            return {"type": "step_start", "id": event.get("id")}

        elif event_type == "reasoning-start":
            self.in_reasoning = True
            self.block_index += 1
            return {"type": "reasoning_start", "index": self.block_index}

        elif event_type == "reasoning-delta":
            delta = event.get("delta", "")
            if delta:
                self.reasoning_buffer.append(delta)
            return {"type": "reasoning_delta", "delta": delta, "index": self.block_index}

        elif event_type == "reasoning-end":
            self.in_reasoning = False
            return {"type": "reasoning_end", "index": self.block_index}

        elif event_type == "text-start":
            self.in_text = True
            self.block_index += 1
            return {"type": "text_start", "index": self.block_index}

        elif event_type == "text-delta":
            delta = event.get("delta", "")
            if delta:
                self.text_buffer.append(delta)
            return {"type": "text_delta", "delta": delta, "index": self.block_index}

        elif event_type == "text-end":
            self.in_text = False
            return {"type": "text_end", "index": self.block_index}

        elif event_type == "finish":
            self.finished = True
            self.response_ended = True  # Mark response as ended
            usage = event.get("usage", {})
            self.output_tokens = usage.get("output_tokens", len("".join(self.text_buffer)) // 4)
            self.input_tokens = usage.get("input_tokens", self.input_tokens)
            return {
                "type": "finish",
                "finish_reason": event.get("finishReason", "end_turn"),
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "total_tokens": self.input_tokens + self.output_tokens
                }
            }

        return {"type": "unknown", "raw": event}

    def get_full_text(self) -> str:
        return "".join(self.text_buffer)

    def get_full_reasoning(self) -> str:
        return "".join(self.reasoning_buffer)

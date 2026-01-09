from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

# Claude/Anthropic types
class ClaudeMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ClaudeTool(BaseModel):
    name: str
    description: Optional[str] = ""
    input_schema: Dict[str, Any]

class ClaudeRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    max_tokens: int = 4096
    temperature: Optional[float] = None
    tools: Optional[List[ClaudeTool]] = None
    stream: bool = False
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    thinking: Optional[Dict[str, Any]] = None

# OpenAI types
class OpenAIMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class ResponsesRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[Dict[str, Any]]]
    instructions: Optional[str] = None
    stream: Optional[bool] = False

# Gemini types
class GeminiPart(BaseModel):
    text: Optional[str] = None

class GeminiContent(BaseModel):
    parts: List[GeminiPart]
    role: Optional[str] = None

class GeminiRequest(BaseModel):
    contents: List[GeminiContent]
    generationConfig: Optional[Dict[str, Any]] = None

# Gumloop internal types
class GumloopMessage(BaseModel):
    id: str
    role: str
    content: Optional[str] = None
    parts: Optional[List[Dict[str, Any]]] = None
    timestamp: Optional[str] = None

class GumloopContext(BaseModel):
    chat: Dict[str, Any]
    type: str = "chat"
    gummie_id: str

class GumloopPayload(BaseModel):
    id_token: str
    context: GumloopContext

"""
Tool conversion utilities for Gumloop.
Converts Claude tool_use/tool_result to text format and parses tool calls from response.
"""
import json
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple

def tools_to_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """Convert tool definitions to system prompt format."""
    if not tools:
        return ""

    lines = ["You have access to the following tools:\n"]
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        schema = tool.get("input_schema", {})

        lines.append(f"<tool name=\"{name}\">")
        if desc:
            lines.append(f"<description>{desc}</description>")
        if schema:
            lines.append(f"<parameters>{json.dumps(schema, ensure_ascii=False)}</parameters>")
        lines.append("</tool>\n")

    lines.append("""
When you need to use a tool, output it in this exact format:
<tool_use>
<name>tool_name</name>
<input>{"param": "value"}</input>
</tool_use>

You can use multiple tools in one response. After outputting tool_use blocks, wait for the tool results before continuing.
""")
    return "\n".join(lines)

def convert_message_content(content: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert message content to text, extracting tool_use and tool_result blocks.
    Returns (text_content, tool_blocks)
    """
    if isinstance(content, str):
        return content, []

    if not isinstance(content, list):
        return str(content), []

    text_parts = []
    tool_blocks = []

    for block in content:
        if not isinstance(block, dict):
            continue

        btype = block.get("type", "")

        if btype == "text":
            text_parts.append(block.get("text", ""))

        elif btype == "tool_use":
            tool_blocks.append({
                "type": "tool_use",
                "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": block.get("name", ""),
                "input": block.get("input", {})
            })

        elif btype == "tool_result":
            tool_blocks.append({
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id", ""),
                "content": block.get("content", ""),
                "is_error": block.get("is_error", False)
            })

    return "\n".join(text_parts), tool_blocks

def tool_result_to_text(tool_result: Dict[str, Any]) -> str:
    """Convert tool_result block to text format."""
    tool_use_id = tool_result.get("tool_use_id", "")
    content = tool_result.get("content", "")
    is_error = tool_result.get("is_error", False)

    if isinstance(content, list):
        # Extract text from content blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        content = "\n".join(text_parts)

    status = "error" if is_error else "success"
    return f"<tool_result tool_use_id=\"{tool_use_id}\" status=\"{status}\">\n{content}\n</tool_result>"

def tool_use_to_text(tool_use: Dict[str, Any]) -> str:
    """Convert tool_use block to text format (for assistant messages in history)."""
    name = tool_use.get("name", "")
    input_data = tool_use.get("input", {})
    tool_id = tool_use.get("id", "")

    input_json = json.dumps(input_data, ensure_ascii=False)
    return f"<tool_use id=\"{tool_id}\">\n<name>{name}</name>\n<input>{input_json}</input>\n</tool_use>"

def merge_consecutive_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive messages with the same role to ensure strict alternation.
    This prevents infinite loops caused by multiple user messages in a row.
    """
    if not messages:
        return []

    result = []
    pending_contents = []
    pending_role = None

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == pending_role:
            # Same role, accumulate content
            if content:
                pending_contents.append(content)
        else:
            # Different role, flush pending
            if pending_role and pending_contents:
                result.append({"role": pending_role, "content": "\n\n".join(pending_contents)})
            pending_role = role
            pending_contents = [content] if content else []

    # Flush remaining
    if pending_role and pending_contents:
        result.append({"role": pending_role, "content": "\n\n".join(pending_contents)})

    return result

def convert_messages_simple(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert messages without embedding tools (tools are set via REST API).
    Only converts tool_use/tool_result blocks to text format.
    """
    result = []
    seen_tool_result_ids = set()

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        text_content, tool_blocks = convert_message_content(content)

        if role == "assistant":
            parts = []
            if text_content:
                parts.append(text_content)
            for block in tool_blocks:
                if block.get("type") == "tool_use":
                    parts.append(tool_use_to_text(block))
            if parts:
                result.append({"role": "assistant", "content": "\n".join(parts)})

        elif role == "user":
            parts = []
            if text_content:
                parts.append(text_content)
            for block in tool_blocks:
                if block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    if tool_use_id and tool_use_id in seen_tool_result_ids:
                        continue
                    if tool_use_id:
                        seen_tool_result_ids.add(tool_use_id)
                    parts.append(tool_result_to_text(block))
            if parts:
                result.append({"role": "user", "content": "\n".join(parts)})

    # Merge consecutive messages with same role to ensure strict alternation
    return merge_consecutive_messages(result)

def convert_messages_with_tools(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    system: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convert messages with tool_use/tool_result to plain text format.
    Includes deduplication of tool_results to prevent infinite loops.
    """
    result = []
    seen_tool_result_ids = set()  # Track processed tool_result IDs

    # Add system prompt with tools
    system_parts = []
    if system:
        system_parts.append(system)
    if tools:
        system_parts.append(tools_to_system_prompt(tools))

    if system_parts:
        result.append({"role": "user", "content": f"[System]: {chr(10).join(system_parts)}"})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        text_content, tool_blocks = convert_message_content(content)

        if role == "assistant":
            # Convert tool_use blocks to text
            parts = []
            if text_content:
                parts.append(text_content)
            for block in tool_blocks:
                if block.get("type") == "tool_use":
                    parts.append(tool_use_to_text(block))

            if parts:
                result.append({"role": "assistant", "content": "\n".join(parts)})

        elif role == "user":
            # Convert tool_result blocks to text, with deduplication
            parts = []
            if text_content:
                parts.append(text_content)
            for block in tool_blocks:
                if block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    # Skip duplicate tool_results
                    if tool_use_id and tool_use_id in seen_tool_result_ids:
                        continue
                    if tool_use_id:
                        seen_tool_result_ids.add(tool_use_id)
                    parts.append(tool_result_to_text(block))

            if parts:
                result.append({"role": "user", "content": "\n".join(parts)})

    # Merge consecutive messages with same role to ensure strict alternation
    return merge_consecutive_messages(result)

# Regex patterns for parsing tool calls
TOOL_USE_PATTERN = re.compile(
    r'<tool_use(?:\s+id="([^"]*)")?>\s*<name>([^<]+)</name>\s*<input>(.*?)</input>\s*</tool_use>',
    re.DOTALL
)

def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse tool calls from model output.
    Returns (remaining_text, tool_use_blocks)
    """
    tool_uses = []

    for match in TOOL_USE_PATTERN.finditer(text):
        tool_id = match.group(1) or f"toolu_{uuid.uuid4().hex[:24]}"
        name = match.group(2).strip()
        input_str = match.group(3).strip()

        try:
            input_data = json.loads(input_str)
        except json.JSONDecodeError:
            input_data = {"raw": input_str}

        tool_uses.append({
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": input_data
        })

    # Remove tool_use blocks from text
    remaining_text = TOOL_USE_PATTERN.sub("", text).strip()

    return remaining_text, tool_uses

def detect_tool_loop(messages: List[Dict[str, Any]], threshold: int = 3) -> Optional[str]:
    """
    Detect if the same tool is being called repeatedly (potential infinite loop).

    Checks for:
    1. Same tool called N times with same input consecutively
    2. Duplicate tool_result IDs across messages
    """
    recent_calls = []
    seen_tool_result_ids = set()
    duplicate_results = []

    for msg in messages[-15:]:  # Check last 15 messages
        content = msg.get("content", "")
        role = msg.get("role", "")

        _, tool_blocks = convert_message_content(content)

        if role == "assistant":
            for block in tool_blocks:
                if block.get("type") == "tool_use":
                    call_sig = (block.get("name"), json.dumps(block.get("input", {}), sort_keys=True))
                    recent_calls.append(call_sig)

        elif role == "user":
            for block in tool_blocks:
                if block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    if tool_use_id:
                        if tool_use_id in seen_tool_result_ids:
                            duplicate_results.append(tool_use_id)
                        seen_tool_result_ids.add(tool_use_id)

    # Check for consecutive identical tool calls
    if len(recent_calls) >= threshold:
        last_call = recent_calls[-1]
        consecutive = sum(1 for c in recent_calls[-threshold:] if c == last_call)
        if consecutive >= threshold:
            return f"Detected infinite loop: tool '{last_call[0]}' called {consecutive} times consecutively with same input"

    # Check for duplicate tool_results (sign of message ordering issue)
    if len(duplicate_results) >= 2:
        return f"Detected duplicate tool_results: {duplicate_results[:3]}. This may cause infinite loops."

    return None

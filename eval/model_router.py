# eval/model_router.py
import os
import json
from typing import Dict, List

# --- map parser: "default=openai;dimension=claude;functional_performance=claude"
def parse_model_map(s: str | None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not s:
        return mapping
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Bad model-map entry: {chunk!r}. Use key=value;key=value")
        k, v = chunk.split("=", 1)
        mapping[k.strip().lower()] = v.strip().lower()
    return mapping

def choose_backend_for_subset(subset: str, model_map: Dict[str, str]) -> str:
    sk = subset.strip().lower()
    if sk in model_map:
        return model_map[sk]
    if "default" in model_map:
        return model_map["default"]
    return "openai"  # safe default

# --- OpenAI chat
def openai_chat(messages: List[dict], model: str | None = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """
    Requires OPENAI_API_KEY. Uses official OpenAI Python client.
    """
    from openai import OpenAI
    client = OpenAI()
    mdl = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=mdl,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

# --- Claude chat (prefers Bedrock if AWS creds/region exist, else Anthropic API)
def claude_chat(messages: List[dict], model: str | None = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    mdl = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")

    if os.getenv("AWS_REGION"):
        import boto3
        bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))

        # Convert OpenAI-style messages to Claude messages
        sys = []
        conv = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                sys.append(content)
            elif role == "user":
                conv.append({"role": "user", "content": [{"type": "text", "text": content}]})
            elif role == "assistant":
                conv.append({"role": "assistant", "content": [{"type": "text", "text": content}]})

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conv,
            "model": mdl,
        }
        if sys:
            body["system"] = "\n".join(sys)

        resp = bedrock.invoke_model(
            modelId=mdl,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        out = json.loads(resp["body"].read())
        return "".join([b.get("text", "") for b in out.get("content", []) if b.get("type") == "text"])

    elif os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        sys = []
        conv = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                sys.append(content)
            elif role == "user":
                conv.append({"role": "user", "content": content})
            elif role == "assistant":
                conv.append({"role": "assistant", "content": content})

        resp = client.messages.create(
            model=mdl,
            max_tokens=max_tokens,
            temperature=temperature,
            system="\n".join(sys) if sys else None,
            messages=conv
        )
        # Flatten text blocks
        parts = []
        for block in resp.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    else:
        raise RuntimeError("Claude credentials not found. Set AWS_REGION (+ AWS creds) or ANTHROPIC_API_KEY.")
# eval/model_router.py
import os
import json
from typing import Dict, List

# --- Parse "default=openai;dimension=anthropic;functional_performance=anthropic"
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
    return (os.getenv("DQ_PROVIDER") or "openai").strip().lower()

# --- Backends ---

def openai_chat(messages: List[dict], model: str | None = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """
    Requires OPENAI_API_KEY. Uses official OpenAI Python client.
    """
    from openai import OpenAI
    client = OpenAI()
    mdl = model or os.getenv("OPENAI_MODEL") or os.getenv("DQ_OPENAI_MODEL") or "gpt-4o-mini"
    resp = client.chat.completions.create(
        model=mdl,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

def claude_chat(messages: List[dict], model: str | None = None, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """
    Uses Anthropic API by default when ANTHROPIC_API_KEY is set; optional AWS Bedrock fallback if AWS creds/region present.
    """
    mdl = model or os.getenv("CLAUDE_MODEL") or os.getenv("DQ_ANTHROPIC_MODEL") or "claude-3-5-sonnet-20240620"

    # Preferred: Anthropic API
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        sys_msgs: List[str] = []
        conv: List[dict] = []
        for m in messages:
            role = m.get("role","")
            content = m.get("content","")
            if role == "system":
                sys_msgs.append(content)
            elif role == "user":
                conv.append({"role":"user","content":content})
            elif role == "assistant":
                conv.append({"role":"assistant","content":content})
        resp = client.messages.create(
            model=mdl,
            max_tokens=max_tokens,
            temperature=temperature,
            system="\n".join(sys_msgs) if sys_msgs else None,
            messages=conv
        )
        parts: List[str] = []
        for block in resp.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text",""))
        return "".join(parts)

    # Optional: AWS Bedrock fallback
    if os.getenv("AWS_REGION"):
        import boto3
        bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
        sys_msgs: List[str] = []
        conv: List[dict] = []
        for m in messages:
            role = m.get("role","")
            content = m.get("content","")
            if role == "system":
                sys_msgs.append(content)
            elif role == "user":
                conv.append({"role":"user","content":[{"type":"text","text":content}]})
            elif role == "assistant":
                conv.append({"role":"assistant","content":[{"type":"text","text":content}]})
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conv,
            "model": mdl,
        }
        if sys_msgs:
            body["system"] = "\n".join(sys_msgs)
        resp = bedrock.invoke_model(
            modelId=mdl,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        out = json.loads(resp["body"].read())
        return "".join([b.get("text","") for b in out.get("content",[]) if b.get("type")=="text"])

    raise RuntimeError("Claude credentials not found. Set ANTHROPIC_API_KEY (or AWS_REGION + AWS creds).")

def mock_chat(messages: List[dict], **kwargs) -> str:
    # Deterministic, token-free response to exercise the pipeline
    last_user = next((m.get("content","") for m in reversed(messages) if m.get("role")=="user"), "")
    return f"[MOCK ANSWER] {last_user[:200]}"
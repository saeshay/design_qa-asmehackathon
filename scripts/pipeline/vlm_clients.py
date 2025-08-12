import os
import base64
from io import BytesIO
from typing import Optional, Tuple
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Pillow for robust image open/resize/format sniffing
from PIL import Image

# ---------- Utilities ----------

MAX_SIDE = 4000  # safety margin well below 8000px provider cap

def _open_and_prepare_image(image_path: str) -> Tuple[str, str]:
    """
    Open an image, downscale if needed, and return (mime, base64_str).

    - Supports PNG/JPEG.
    - Enforces that max(width, height) <= MAX_SIDE.
    - Preserves original format if jpg/png; otherwise converts to PNG.
    """
    if not image_path:
        raise ValueError("image_path not provided")

    with Image.open(image_path) as im:
        im = im.convert("RGB") if im.mode not in ("RGB", "RGBA", "L") else im

        # downscale if any side too large
        w, h = im.size
        max_dim = max(w, h)
        if max_dim > MAX_SIDE:
            scale = MAX_SIDE / float(max_dim)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)

        # decide mime/format
        fmt = (im.format or "").upper()
        # If Pillow lost format after conversion, infer from path suffix
        ext = os.path.splitext(image_path)[1].lower()
        # Prefer PNG for unknown/ext-less content
        if fmt in ("PNG", "JPEG", "JPG"):
            out_fmt = "PNG" if fmt == "PNG" else "JPEG"
        else:
            if ext in (".png",):
                out_fmt = "PNG"
            elif ext in (".jpg", ".jpeg"):
                out_fmt = "JPEG"
            else:
                out_fmt = "PNG"

        mime = "image/png" if out_fmt == "PNG" else "image/jpeg"

        # re-encode into memory in canonical format
        buf = BytesIO()
        save_kwargs = {}
        if out_fmt == "JPEG":
            # reasonable quality for diagrams, reduces payload size
            save_kwargs.update(dict(quality=92, optimize=True))
        im.save(buf, format=out_fmt, **save_kwargs)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return mime, b64


# ---------- Mock (keeps repo runnable for $0) ----------

class MockClient:
    def __init__(self, model: str = "mock"):
        self.model = model

    def answer(self, prompt: str, image_path: Optional[str] = None, max_tokens: int = 64) -> str:
        # deterministic, zero-cost placeholder
        # keep response shape consistent with subset prompts
        if "Answer:" in prompt:
            return "Explanation: mock\nAnswer: insufficient"
        return "insufficient"


# ---------- OpenAI (vision + text) ----------

class OpenAIClient:
    def __init__(self, model: str = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("DQ_OPENAI_MODEL", "gpt-4o-mini")

    @retry(
        wait=wait_exponential(min=0.5, max=12),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def answer(self, prompt: str, image_path: Optional[str] = None, max_tokens: int = 64) -> str:
        content = [{"type": "text", "text": prompt}]
        if image_path:
            mime, b64 = _open_and_prepare_image(image_path)
            # NOTE: OpenAI chat.completions wants type="image_url" with a data URL
            content.append(
                {
                    "type": "image_url",
                    "image_url": f"data:{mime};base64,{b64}",
                }
            )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt


# ---------- Anthropic (vision + text) ----------

class AnthropicClient:
    def __init__(self, model: str = None):
        import anthropic
        self.ant = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model or os.getenv("DQ_ANTHROPIC_MODEL", "claude-3-haiku-20240307")

    @retry(
        wait=wait_exponential(min=0.5, max=12),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def answer(self, prompt: str, image_path: Optional[str] = None, max_tokens: int = 128) -> str:
        content = [{"type": "text", "text": prompt}]
        if image_path:
            mime, b64 = _open_and_prepare_image(image_path)
            # NOTE: Anthropic wants a structured base64 image block
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,  # "image/png" or "image/jpeg"
                        "data": b64,
                    },
                }
            )

        msg = self.ant.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )

        # Pull first text block
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
                return block.text.strip()
        # Some SDK versions return msg.content as list[dict]
        if isinstance(getattr(msg, "content", None), list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    return block["text"].strip()
        return ""


# ---------- Factory ----------

def make_client():
    """
    Picks a client based on env:
    - DQ_PROVIDER in {"openai", "anthropic", "mock"} (default "mock")
    - DQ_OPENAI_MODEL, DQ_ANTHROPIC_MODEL override defaults
    """
    provider = os.getenv("DQ_PROVIDER", "mock").strip().lower()
    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    return MockClient()
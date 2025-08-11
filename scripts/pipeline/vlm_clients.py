import os, base64
from tenacity import retry, wait_exponential, stop_after_attempt

# -------- Mock (keeps repo runnable for $0) ----------
class MockClient:
    def __init__(self, model="mock"):
        self.model = model

    def answer(self, prompt, image_path=None, max_tokens=64):
        # super-minimal mock; returns tiny, deterministic tokens
        if "Answer:" in prompt:
            return "Answer: insufficient"
        return "insufficient"

# -------- OpenAI (vision + text) ----------
class OpenAIClient:
    def __init__(self, model="gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
    def answer(self, prompt, image_path=None, max_tokens=64):
        content = [{"type": "text", "text": prompt}]
        if image_path:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

# -------- Anthropic (vision + text) ----------
class AnthropicClient:
    def __init__(self, model="claude-3-haiku-20240307"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
    def answer(self, prompt, image_path=None, max_tokens=128):
        content = [{"type": "text", "text": prompt}]
        if image_path:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        # anthropic returns list of content blocks; pick first text block
        for block in msg.content:
            if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
                return block.text.strip()
        return ""

# -------- Additional VLM helpers (drop-in) ----------
import io
from PIL import Image

def _prepare_image(image_path: str, max_side: int = None, prefer_jpeg: bool = True):
    # Downscale so max(width, height) <= max_side (default from env DQ_IMG_MAX_SIDE=3072)
    # Convert to JPEG if needed, quality ~85, return (mime, base64 string)
    max_side = max_side or int(os.getenv("DQ_IMG_MAX_SIDE", "3072"))
    ext = os.path.splitext(image_path)[-1].lower()
    im = Image.open(image_path)
    if max(im.size) > max_side:
        ratio = max_side / max(im.size)
        new_size = (int(im.size[0] * ratio), int(im.size[1] * ratio))
        im = im.resize(new_size, Image.LANCZOS)
    mime = "image/png" if ext in [".png"] else "image/jpeg"
    if prefer_jpeg or mime == "image/png":
        with io.BytesIO() as buf:
            im.convert("RGB").save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
        mime = "image/jpeg"
    else:
        with io.BytesIO() as buf:
            im.save(buf, format="PNG")
            data = buf.getvalue()
    b64 = base64.b64encode(data).decode()
    return mime, b64

class OpenAIVLMClient:
    def __init__(self, api_key):
        self.api_key = api_key
    def format_payload(self, image_path):
        mime, b64 = _prepare_image(image_path)
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

class AnthropicVLMClient:
    def __init__(self, api_key):
        self.api_key = api_key
    def format_payload(self, image_path):
        mime, b64 = _prepare_image(image_path)
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}

class MockVLMClient:
    def format_payload(self, image_path):
        return {"type": "mock", "path": image_path}
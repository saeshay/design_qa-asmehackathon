import os
import base64
import mimetypes
from io import BytesIO
from tenacity import retry, wait_exponential, stop_after_attempt
from PIL import Image

# -------- Mock (keeps repo runnable for $0) ----------
class MockClient:
    def __init__(self, model="mock"):
        self.model = model

    def answer(self, prompt, image_path=None, max_tokens=64):
        if "Answer:" in prompt:
            return "Answer: insufficient"
        return "insufficient"

# ---------- helpers: resize + base64 (returns mime, b64) ----------
def _prepare_image(image_path: str, max_side: int = None, prefer_jpeg: bool = True):
    """
    Downscale so max(width,height) <= max_side (env DQ_IMG_MAX_SIDE or 3072),
    convert to JPEG if requested, return (mime, base64_str).
    """
    max_side = max_side or int(os.getenv("DQ_IMG_MAX_SIDE", "3072"))
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) > max_side:
        scale = float(max_side) / float(max(w, h))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    use_jpeg = prefer_jpeg
    if use_jpeg and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    buf = BytesIO()
    if use_jpeg:
        img.save(buf, format="JPEG", quality=85, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG")
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return mime, b64

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
            mime, b64 = _prepare_image(image_path, max_side=int(os.getenv("DQ_IMG_MAX_SIDE", "3072")))
            data_url = f"data:{mime};base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
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
            # Use resized image and the correct MIME to avoid media_type mismatch.
            mime, b64 = _prepare_image(image_path, max_side=int(os.getenv("DQ_IMG_MAX_SIDE", "3072")))
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64}
            })
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        for block in msg.content:
            if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
                return block.text.strip()
        return ""
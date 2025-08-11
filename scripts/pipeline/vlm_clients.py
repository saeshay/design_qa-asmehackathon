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
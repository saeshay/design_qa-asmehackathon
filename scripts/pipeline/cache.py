import os, json, hashlib
CACHE = "your_outputs/.cache.jsonl"
os.makedirs(os.path.dirname(CACHE), exist_ok=True)

def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def make_key(subset, qid, provider, model, prompt):
    return f"{subset}:{qid}:{provider}:{model}:{_h(prompt)}"

def get(key):
    if not os.path.exists(CACHE): return None
    with open(CACHE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("key") == key: return obj.get("value")
            except Exception:
                pass
    return None

def put(key, value):
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")
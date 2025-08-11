import re
from typing import Optional

MAX_LEN = 120


def clamp_len(s: str, max_len: int = MAX_LEN) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[:max_len].rstrip()


def normalize_yes_no(s: str) -> str:
    s = s or ""
    s = s.strip()
    # Extract trailing yes/no after Answer: if present
    lower = s.lower()
    if "answer:" in lower:
        tail = lower.split("answer:")[-1].strip()
        token = tail.split()[0] if tail else ""
    else:
        token = lower.split()[0] if lower else ""
    if token in {"yes", "no"}:
        return token
    return "INSUFFICIENT"


def normalize_rule_text(s: str) -> str:
    s = s or ""
    # Remove leading labels
    s = re.sub(r"^\s*(answer:|rule:|text:)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return clamp_len(s)


def normalize_rule_list(s: str) -> str:
    s = s or ""
    # Accept Python list string or comma-separated
    if s.strip().startswith("[") and s.strip().endswith("]"):
        # Extract items
        items = re.findall(r"([A-Z]{1,3}\.\d+(?:\.\d+){0,3})", s)
    else:
        items = [t.strip() for t in s.split(',') if t.strip()]
    if not items:
        return "INSUFFICIENT"
    # Deduplicate, keep order
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return clamp_len(", ".join(out))


def normalize_component_name(s: str) -> str:
    s = s or ""
    # Just keep short noun phrase
    s = re.sub(r"^\s*(answer:|component:)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "INSUFFICIENT"
    return clamp_len(s)


def normalize_explained_yes_no(s: str) -> str:
    s = s or ""
    # Expect "Explanation: ... Answer: yes/no"
    lower = s.lower()
    # Ensure we have Answer:
    if "answer:" not in lower:
        # Try to recover if a yes/no exists
        yn = normalize_yes_no(s)
        if yn == "INSUFFICIENT":
            return "INSUFFICIENT"
        return f"Explanation: {clamp_len(s)}\nAnswer: {yn}"
    # Trim to only one Answer and one Explanation
    # Keep the first Explanation: and the last Answer:
    parts = re.split(r"(explanation:|answer:)", s, flags=re.IGNORECASE)
    # Reconstruct normalized order
    explanation_text = None
    answer_text = None
    i = 0
    while i < len(parts) - 1:
        tag = parts[i].lower()
        content = parts[i + 1]
        if tag == "explanation:":
            explanation_text = content.strip()
        if tag == "answer:":
            answer_text = content.strip()
        i += 2
    yn = normalize_yes_no(answer_text or "")
    if yn == "INSUFFICIENT":
        return "INSUFFICIENT"
    explanation_text = explanation_text or ""
    explanation_text = clamp_len(explanation_text)
    return f"Explanation: {explanation_text}\nAnswer: {yn}"


def is_short_phrase(s: str, max_words: int = 4) -> bool:
    words = re.findall(r"[A-Za-z0-9\-]+", s or "")
    return 1 <= len(words) <= max_words
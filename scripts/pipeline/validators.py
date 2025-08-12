import re

YES_NO = {"yes", "no"}

# Existing helpers (kept for compatibility)
_RULE_RX = re.compile(r"\b[A-Z]{1,3}\.\d+(?:\.\d+)*\b")

def norm(s: str) -> str:
    return (s or "").strip()

def is_yes_no(s: str) -> bool:
    s = norm(s).lower()
    return s == "yes" or s == "no"

def force_yes_no(s: str) -> str:
    s = norm(s).lower()
    if "yes" in s:
        return "yes"
    if "no" in s:
        return "no"
    return ""

def is_short_phrase(s: str, max_words=4, max_len=60) -> bool:
    s = norm(s)
    if not s or len(s) > max_len:
        return False
    words = re.findall(r"[A-Za-z0-9\-]+", s)
    return 1 <= len(words) <= max_words

def extract_rule_ids(s: str):
    return _RULE_RX.findall(s or "")

def has_expl_and_answer(s: str) -> bool:
    s = norm(s).lower()
    return ("explanation:" in s) and ("answer:" in s)

def good_compilation(s: str, min_rules=1) -> bool:
    rules = extract_rule_ids(s)
    return len(set(rules)) >= min_rules

def good_retrieval(s: str, min_len=20) -> bool:
    s = norm(s)
    return len(s) >= min_len and not _RULE_RX.search(s)

def limit_ocr(s: str, limit=200) -> str:
    return (s or "")[:limit]

# Additional strict helpers per spec

def normalize_yes_no_block(text: str) -> str:
    """
    Returns canonical two-line template if we can find a yes/no.
    Prefers an explicit `Answer:` line; otherwise falls back to any yes/no token.
    If nothing clear, returns the original text unchanged.
    """
    if not isinstance(text, str):
        return text
    s = text.strip()
    m = re.search(r'(?im)^\s*answer\s*:\s*([^\n\r]+)', s)
    if m:
        v = m.group(1).strip().lower()
        v = "yes" if "yes" in v else ("no" if "no" in v else None)
    else:
        m2 = re.search(r'(?i)\b(yes|no)\b', s)
        v = m2.group(1).lower() if m2 else None
    if v in YES_NO:
        exp = None
        mexp = re.search(r'(?im)^\s*explanation\s*:\s*(.+)$', s)
        if mexp:
            exp = mexp.group(1).strip()
        return f"Explanation: {exp or '—'}\nAnswer: {v}"
    return s

def is_yes_no_block(text: str) -> bool:
    if not isinstance(text, str):
        return False
    m = re.search(r'(?im)^\s*answer\s*:\s*([^\n\r]+)', text or "")
    if m:
        v = m.group(1).strip().lower()
        return ("yes" in v) or ("no" in v)
    return bool(re.search(r'(?i)\b(yes|no)\b', text or ""))

def presence_strict(text: str) -> str:
    if not isinstance(text, str):
        return "INSUFFICIENT"
    m = re.search(r'(?i)\b(yes|no)\b', text)
    return m.group(1).lower() if m else "INSUFFICIENT"

# New helpers for unified block formatting

def build_block(explanation: str, answer_yes_no: str) -> str:
    ans = answer_yes_no.strip().lower()
    if ans not in YES_NO:
        ans = "no"
    exp = (explanation or "").strip() or "—"
    return f"Explanation: {exp}\nAnswer: {ans}"


def retrieval_nontrivial(s: str, min_len: int = 20) -> bool:
    if not isinstance(s, str):
        return False
    txt = s.strip().lower()
    return len(txt) >= min_len and "insufficient" not in txt
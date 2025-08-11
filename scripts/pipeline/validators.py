import re

_RULE_RX = re.compile(r"\b[A-Z]{1,3}\.\d+(?:\.\d+)*\b")  # e.g., T.7.1.2 or V.1


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
    # Retrieval expects verbatim rule text; we just enforce it's non-trivial text.
    s = norm(s)
    return len(s) >= min_len and not _RULE_RX.search(s)  # text not just rule ids


def limit_ocr(s: str, limit=200) -> str:
    return (s or "")[:limit]
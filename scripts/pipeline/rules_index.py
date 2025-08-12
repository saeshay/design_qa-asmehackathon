import os
import re
from typing import Dict, Optional

RULE_ID_RX = re.compile(r"^([A-Z]{1,3}\.\d+(?:\.\d+)*)\b")


class RulesIndex:
    """Deterministic index over a text dump of the FSAE rules.

    The input is a plain-text file where each rule begins with a rule id
    at the start of a line, e.g., "V.1" or "T.7.1.2".

    We capture the subsequent lines until the next rule id line.
    The value returned is a single-line text with normalized spaces.
    """

    def __init__(self, rules_txt_path: Optional[str] = None):
        self._rules_txt_path = (
            rules_txt_path
            or os.getenv("DQ_RULES_TXT")
            or self._first_existing([
                "dataset/rules/rules_2024.txt",
                "dataset/docs/rules_pdfplumber1.txt",
            ])
        )
        if not self._rules_txt_path or not os.path.exists(self._rules_txt_path):
            raise FileNotFoundError(
                f"Rules text not found. Set DQ_RULES_TXT or place file at dataset/rules/rules_2024.txt. Tried: {self._rules_txt_path}"
            )
        self._index: Dict[str, str] = {}
        self._loaded = False
        self._load()

    @staticmethod
    def _first_existing(paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        # collapse whitespace and join into a single line
        return " ".join((text or "").split()).strip().strip('"')

    def _load(self):
        if self._loaded:
            return
        current_id = None
        buffer = []
        with open(self._rules_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")
                m = RULE_ID_RX.match(line.strip())
                if m:
                    # flush previous
                    if current_id is not None:
                        self._index[current_id] = self._normalize("\n".join(buffer))
                    # start new rule
                    current_id = m.group(1)
                    # remove the id from the line, keep remainder as start of text
                    remainder = line[len(m.group(1)) :].strip(" -:\t")
                    buffer = [remainder] if remainder else []
                else:
                    # accumulate
                    if current_id is not None:
                        buffer.append(line)
        # flush last
        if current_id is not None:
            self._index[current_id] = self._normalize("\n".join(buffer))
        self._loaded = True

    def get(self, rule_id: str) -> Optional[str]:
        if not rule_id:
            return None
        return self._index.get(rule_id)
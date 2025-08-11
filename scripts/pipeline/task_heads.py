import ast
from dataclasses import dataclass
from typing import Optional, List

import os
import pandas as pd

from .format_checker import (
    normalize_yes_no,
    normalize_rule_text,
    normalize_rule_list,
    normalize_component_name,
    normalize_explained_yes_no,
    is_short_phrase,
)
from .rule_retriever import RuleAwareRetriever
from .cache import make_key, get as cache_get, put as cache_put
from .validators import (
    force_yes_no as v_force_yes_no,
    is_yes_no as v_is_yes_no,
    is_short_phrase as v_is_short_phrase,
    good_compilation as v_good_compilation,
    has_expl_and_answer as v_has_expl_and_answer,
    good_retrieval as v_good_retrieval,
    extract_rule_ids as v_extract_rule_ids,
)


@dataclass
class SubsetPrompts:
    retrieval_system: str
    compilation_system: str
    definition_system: str
    presence_system: str
    dimension_system: str
    functional_system: str


def _client_meta(client) -> tuple:
    provider = getattr(client, "__class__", type(client)).__name__.replace("Client", "").lower() or "unknown"
    model = getattr(client, "model", "") or ""
    return provider, model


class EscalationBudget:
    """Controls how many items we may escalate. Defaults: <=10% or explicit cap."""
    def __init__(self, total_items: int):
        self.total = max(1, int(total_items))
        cap_pct = float(os.getenv("DQ_ESC_PCT_MAX", "0.10"))
        cap_abs = os.getenv("DQ_ESC_ABS_MAX")
        self.max_escalations = int(cap_abs) if cap_abs else int(self.total * cap_pct + 0.5)
        self.used = 0

    def allow(self) -> bool:
        return self.used < self.max_escalations

    def consume(self):
        self.used += 1


def _ask_with_cache(client, subset, qid, prompt, image_path=None, max_tokens=64):
    provider_env = os.getenv("DQ_PROVIDER", "mock")
    model = getattr(client, "model", "unknown")
    key = make_key(subset, qid, provider_env, model, prompt + (image_path or ""))
    cached = cache_get(key)
    if cached is not None:
        return cached
    ans = client.answer(prompt, image_path=image_path, max_tokens=max_tokens)
    cache_put(key, ans)
    return ans


class RetrievalHead:
    def __init__(self, client, retriever: RuleAwareRetriever, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, out_csv: str, limit: Optional[int] = None):
        df = pd.read_csv(in_csv)
        if limit is not None:
            df = df.head(limit)
        budget = EscalationBudget(len(df))
        preds = []
        for idx, row in df.iterrows():
            context_snips = []
            try:
                context_snips = self.retriever.get_top_k_snippets(row['question'], k=2, char_limit=200)
            except Exception:
                context_snips = []
            ctx = ("\n" + "\n".join(context_snips)) if context_snips else ""
            q = f"{self.system_prompt}\n{row['question']}{ctx}"
            # Attempt exact lookup to produce verbatim rule text
            rn = None
            for token in row['question'].split():
                if token.count('.') >= 1 and token[0].isalpha():
                    rn = token.strip(',.?:;')
                    break
            verbatim = self.retriever.get_rule_text(rn) if rn else None
            if verbatim:
                raw = verbatim
                preds.append(normalize_rule_text(raw))
                continue
            raw = _ask_with_cache(self.client, "retrieval", idx, q, max_tokens=64)
            norm = normalize_rule_text(raw)
            if not v_good_retrieval(norm) and self.escalation and budget.allow():
                raw2 = _ask_with_cache(self.escalation, "retrieval", idx, q, max_tokens=64)
                norm2 = normalize_rule_text(raw2)
                if v_good_retrieval(norm2):
                    budget.consume()
                    preds.append(norm2)
                    continue
            preds.append(norm if v_good_retrieval(norm) else "INSUFFICIENT")
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class CompilationHead:
    def __init__(self, client, retriever: RuleAwareRetriever, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, out_csv: str, limit: Optional[int] = None):
        df = pd.read_csv(in_csv)
        if limit is not None:
            df = df.head(limit)
        budget = EscalationBudget(len(df))
        preds = []
        for idx, row in df.iterrows():
            # Parse term inside backticks
            term = row['question'].split('`')
            term = term[1] if len(term) >= 2 else row['question']
            rules = self.retriever.find_relevant_rules(term)
            raw = ", ".join(rules)
            if not v_good_compilation(raw) and self.client:
                q = (
                    "List only relevant rule numbers separated by commas (e.g., 'T.7.1, T.7.2'). No extra words.\n"
                    f"Question: {row['question']}\n"
                    "Answer:"
                )
                raw2 = _ask_with_cache(self.client, "compilation", idx, q, max_tokens=64)
                if not v_good_compilation(raw2) and self.escalation and budget.allow():
                    raw3 = _ask_with_cache(self.escalation, "compilation", idx, q, max_tokens=64)
                    if v_good_compilation(raw3):
                        budget.consume()
                        raw = raw3
                else:
                    raw = raw2
            preds.append(normalize_rule_list(raw) if v_good_compilation(raw) else "")
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class DefinitionHead:
    def __init__(self, client, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str, limit: Optional[int] = None):
        df = pd.read_csv(in_csv)
        if limit is not None:
            df = df.head(limit)
        budget = EscalationBudget(len(df))
        preds = []
        for idx, row in df.iterrows():
            ctx = ""
            q = (
                "Identify the highlighted CAD component; return a short noun phrase only.\n"
                f"Question: {row['question']}\n"
                "Answer:"
            )
            img = f"{images_dir}/{row['image']}"
            ans = _ask_with_cache(self.client, "definition", idx, q, image_path=img, max_tokens=16).strip()
            if not v_is_short_phrase(ans) and self.escalation and budget.allow():
                ans2 = _ask_with_cache(self.escalation, "definition", idx, q, image_path=img, max_tokens=16).strip()
                if v_is_short_phrase(ans2):
                    budget.consume()
                    preds.append(normalize_component_name(ans2))
                    continue
            preds.append(normalize_component_name(ans if v_is_short_phrase(ans) else "INSUFFICIENT"))
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class PresenceHead:
    def __init__(self, client, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str, limit: Optional[int] = None):
        df = pd.read_csv(in_csv)
        if limit is not None:
            df = df.head(limit)
        budget = EscalationBudget(len(df))
        preds = []
        for idx, row in df.iterrows():
            q = (
                "Answer strictly with 'yes' or 'no'.\n"
                f"Question: {row['question']}\n"
                "Answer:"
            )
            img = f"{images_dir}/{row['image']}"
            ans = v_force_yes_no(_ask_with_cache(self.client, "presence", idx, q, image_path=img, max_tokens=32))
            if not v_is_yes_no(ans) and self.escalation and budget.allow():
                ans2 = v_force_yes_no(_ask_with_cache(self.escalation, "presence", idx, q, image_path=img, max_tokens=32))
                if v_is_yes_no(ans2):
                    budget.consume()
                    preds.append(ans2)
                    continue
            preds.append(ans if v_is_yes_no(ans) else "no")
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class DimensionHead:
    def __init__(self, client, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str, limit: Optional[int] = None):
        df = pd.read_csv(in_csv)
        if limit is not None:
            df = df.head(limit)
        budget = EscalationBudget(len(df))
        preds = []
        for idx, row in df.iterrows():
            q = (
                "Use the drawing to judge compliance. Respond as:\n"
                "Explanation: <one short sentence>\n"
                "Answer: yes/no\n"
                f"Question: {row['question']}\n"
                "Explanation:"
            )
            img = f"{images_dir}/{row['image']}"
            ans = _ask_with_cache(self.client, "dimension", idx, q, image_path=img, max_tokens=96)
            ok = v_has_expl_and_answer(ans) and v_is_yes_no(v_force_yes_no(ans))
            if not ok and self.escalation and budget.allow():
                ans2 = _ask_with_cache(self.escalation, "dimension", idx, q, image_path=img, max_tokens=96)
                ok2 = v_has_expl_and_answer(ans2) and v_is_yes_no(v_force_yes_no(ans2))
                if ok2:
                    budget.consume()
                    preds.append(normalize_explained_yes_no(ans2))
                    continue
            preds.append(normalize_explained_yes_no(ans) if ok else "Explanation: insufficient\nAnswer: no")
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class FunctionalHead:
    def __init__(self, client, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str, limit: Optional[int] = None):
        df = pd.read_csv(in_csv)
        if limit is not None:
            df = df.head(limit)
        budget = EscalationBudget(len(df))
        preds = []
        for idx, row in df.iterrows():
            q = (
                "Use the image and rule to judge compliance. Respond as:\n"
                "Explanation: <one short sentence>\n"
                "Answer: yes/no\n"
                f"Question: {row['question']}\n"
                "Explanation:"
            )
            img = f"{images_dir}/{row['image']}"
            ans = _ask_with_cache(self.client, "functional", idx, q, image_path=img, max_tokens=96)
            ok = v_has_expl_and_answer(ans) and v_is_yes_no(v_force_yes_no(ans))
            if not ok and self.escalation and budget.allow():
                ans2 = _ask_with_cache(self.escalation, "functional", idx, q, image_path=img, max_tokens=96)
                ok2 = v_has_expl_and_answer(ans2) and v_is_yes_no(v_force_yes_no(ans2))
                if ok2:
                    budget.consume()
                    preds.append(normalize_explained_yes_no(ans2))
                    continue
            preds.append(normalize_explained_yes_no(ans) if ok else "Explanation: insufficient\nAnswer: no")
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)
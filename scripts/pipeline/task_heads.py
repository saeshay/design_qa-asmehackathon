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

    def run(self, in_csv: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            q = f"{self.system_prompt}\n{row['question']}"
            # Attempt exact lookup to produce verbatim rule text
            rn = None
            for token in row['question'].split():
                if token.count('.') >= 1 and token[0].isalpha():
                    rn = token.strip(',.?:;')
                    break
            verbatim = self.retriever.get_rule_text(rn) if rn else None
            if verbatim:
                raw = verbatim
            else:
                raw = _ask_with_cache(self.client, "retrieval", idx, q, max_tokens=64)
                norm = normalize_rule_text(raw)
                if (norm == "INSUFFICIENT" or not norm) and self.escalation:
                    raw2 = _ask_with_cache(self.escalation, "retrieval", idx, q, max_tokens=64)
                    norm = normalize_rule_text(raw2)
                preds.append(norm)
                continue
            preds.append(normalize_rule_text(raw))
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

    def run(self, in_csv: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            # Parse term inside backticks
            term = row['question'].split('`')
            term = term[1] if len(term) >= 2 else row['question']
            rules = self.retriever.find_relevant_rules(term)
            raw = ", ".join(rules)
            norm = normalize_rule_list(raw)
            if (norm == "INSUFFICIENT" or not norm) and self.client:
                q = f"List only the relevant rule numbers separated by commas, no extra words.\nQuestion: {row['question']}\nAnswer: "
                raw2 = _ask_with_cache(self.client, "compilation", idx, q, max_tokens=64)
                norm = normalize_rule_list(raw2)
                if (norm == "INSUFFICIENT" or not norm) and self.escalation:
                    raw3 = _ask_with_cache(self.escalation, "compilation", idx, q, max_tokens=64)
                    norm = normalize_rule_list(raw3)
            preds.append(norm if norm != "INSUFFICIENT" else "")
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class DefinitionHead:
    def __init__(self, client, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            q = f"{self.system_prompt}\n{row['question']}"
            img = f"{images_dir}/{row['image']}"
            raw = _ask_with_cache(self.client, "definition", idx, q, image_path=img, max_tokens=16)
            norm = normalize_component_name(raw)
            if (norm == "INSUFFICIENT" or not is_short_phrase(norm)) and self.escalation:
                raw2 = _ask_with_cache(self.escalation, "definition", idx, q, image_path=img, max_tokens=16)
                norm = normalize_component_name(raw2)
            preds.append(norm)
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class PresenceHead:
    def __init__(self, client, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            q = f"You are a design reviewer. Answer strictly yes or no.\nQuestion: {row['question']}\nAnswer: "
            img = f"{images_dir}/{row['image']}"
            raw = _ask_with_cache(self.client, "presence", idx, q, image_path=img, max_tokens=32)
            norm = normalize_yes_no(raw)
            if norm == "INSUFFICIENT" and self.escalation:
                raw2 = _ask_with_cache(self.escalation, "presence", idx, q, image_path=img, max_tokens=32)
                norm = normalize_yes_no(raw2)
            preds.append(norm if norm in {"yes", "no"} else "no")
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class DimensionHead:
    def __init__(self, client, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            q = f"{self.system_prompt}\n{row['question']}"
            img = f"{images_dir}/{row['image']}"
            raw = _ask_with_cache(self.client, "dimension", idx, q, image_path=img, max_tokens=96)
            norm = normalize_explained_yes_no(raw)
            if norm == "INSUFFICIENT" and self.escalation:
                raw2 = _ask_with_cache(self.escalation, "dimension", idx, q, image_path=img, max_tokens=96)
                norm = normalize_explained_yes_no(raw2)
            preds.append(norm)
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class FunctionalHead:
    def __init__(self, client, system_prompt: str, escalation=None):
        self.client = client
        self.escalation = escalation
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            q = f"{self.system_prompt}\n{row['question']}"
            img = f"{images_dir}/{row['image']}"
            raw = _ask_with_cache(self.client, "functional", idx, q, image_path=img, max_tokens=96)
            norm = normalize_explained_yes_no(raw)
            if norm == "INSUFFICIENT" and self.escalation:
                raw2 = _ask_with_cache(self.escalation, "functional", idx, q, image_path=img, max_tokens=96)
                norm = normalize_explained_yes_no(raw2)
            preds.append(norm)
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)
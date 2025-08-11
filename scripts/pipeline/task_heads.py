import ast
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

from .format_checker import (
    normalize_yes_no,
    normalize_rule_text,
    normalize_rule_list,
    normalize_component_name,
    normalize_explained_yes_no,
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


class RetrievalHead:
    def __init__(self, client, retriever: RuleAwareRetriever, system_prompt: str):
        self.client = client
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
                key = make_key("retrieval", idx, self.provider, self.model, q)
                cached = cache_get(key)
                if cached is not None:
                    raw = cached
                else:
                    raw = self.client.answer(q, image_path=None, max_tokens=96)
                    cache_put(key, raw)
            preds.append(normalize_rule_text(raw))
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class CompilationHead:
    def __init__(self, client, retriever: RuleAwareRetriever, system_prompt: str):
        self.client = client
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
            preds.append(normalize_rule_list(raw))
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class DefinitionHead:
    def __init__(self, client, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            q = f"{self.system_prompt}\n{row['question']}"
            img = f"{images_dir}/{row['image']}"
            key = make_key("definition", idx, self.provider, self.model, q + f"|{img}")
            cached = cache_get(key)
            if cached is not None:
                raw = cached
            else:
                raw = self.client.answer(q, image_path=img, max_tokens=16)
                cache_put(key, raw)
            preds.append(normalize_component_name(raw))
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class PresenceHead:
    def __init__(self, client, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            q = f"{self.system_prompt}\n{row['question']}"
            img = f"{images_dir}/{row['image']}"
            key = make_key("presence", idx, self.provider, self.model, q + f"|{img}")
            cached = cache_get(key)
            if cached is not None:
                raw = cached
            else:
                raw = self.client.answer(q, image_path=img, max_tokens=4)
                cache_put(key, raw)
            preds.append(normalize_yes_no(raw))
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class DimensionHead:
    def __init__(self, client, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            q = f"{self.system_prompt}\n{row['question']}"
            img = f"{images_dir}/{row['image']}"
            key = make_key("dimension", idx, self.provider, self.model, q + f"|{img}")
            cached = cache_get(key)
            if cached is not None:
                raw = cached
            else:
                raw = self.client.answer(q, image_path=img, max_tokens=64)
                cache_put(key, raw)
            preds.append(normalize_explained_yes_no(raw))
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)


class FunctionalHead:
    def __init__(self, client, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt
        self.provider, self.model = _client_meta(client)

    def run(self, in_csv: str, images_dir: str, out_csv: str):
        df = pd.read_csv(in_csv)
        preds = []
        for idx, row in df.iterrows():
            q = f"{self.system_prompt}\n{row['question']}"
            img = f"{images_dir}/{row['image']}"
            key = make_key("functional", idx, self.provider, self.model, q + f"|{img}")
            cached = cache_get(key)
            if cached is not None:
                raw = cached
            else:
                raw = self.client.answer(q, image_path=img, max_tokens=64)
                cache_put(key, raw)
            preds.append(normalize_explained_yes_no(raw))
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(out_csv, index=False)
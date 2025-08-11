import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None


@dataclass
class RetrieverConfig:
    rules_csv_path: str = "dataset/docs/csv_rules/all_rules_extracted.csv"
    min_rule_len: int = 40
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    bm25_weight: float = 0.6
    dense_weight: float = 0.4
    top_k: int = 2
    whitelist_terms: Optional[List[str]] = None
    blacklist_terms: Optional[List[str]] = None


class RuleAwareRetriever:
    """
    Hybrid retriever over the FSAE rules with:
    - Exact rule lookup by `rule_num`
    - BM25 over `rule_text`
    - Dense semantic similarity via SentenceTransformer
    - Regex extraction of referenced rules within rule text
    - Term whitelist/blacklist filtering
    """

    def __init__(self, config: RetrieverConfig):
        self.config = config
        self.rules_df = pd.read_csv(config.rules_csv_path, encoding="utf-8-sig")
        # Drop short title rows if needed
        self.rules_df = self.rules_df[self.rules_df["rule_text"].fillna("").str.len() >= config.min_rule_len]
        self.rules_df = self.rules_df.dropna(subset=["rule_num", "rule_text"]).reset_index(drop=True)

        # Caches
        self.rule_num_to_text: Dict[str, str] = {
            str(rnum): rtext for rnum, rtext in zip(self.rules_df["rule_num"], self.rules_df["rule_text"])
        }

        # BM25 index
        self._bm25 = None
        self._tokenized_corpus: List[List[str]] = []
        if BM25Okapi is not None:
            self._tokenized_corpus = [self._tokenize(text) for text in self.rules_df["rule_text"].tolist()]
            self._bm25 = BM25Okapi(self._tokenized_corpus)

        # Dense index
        self._dense_model = None
        self._dense_embeddings = None
        if SentenceTransformer is not None:
            try:
                self._dense_model = SentenceTransformer(self.config.dense_model_name)
                self._dense_embeddings = self._dense_model.encode(
                    self.rules_df["rule_text"].tolist(), batch_size=128, convert_to_tensor=True, show_progress_bar=False
                )
            except Exception:
                self._dense_model = None
                self._dense_embeddings = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _extract_rule_numbers(text: str) -> List[str]:
        # Matches patterns like A.1, AA.1.2, EV.5.9.1 etc.
        pattern = r"\b([A-Z]{1,3}\.\d+(?:\.\d+){0,3})\b"
        return list({m.group(1) for m in re.finditer(pattern, text)})

    def _apply_filters(self, candidates: pd.DataFrame, query: str) -> pd.DataFrame:
        # Whitelist: require at least one term to appear in either query or rule text
        if self.config.whitelist_terms:
            wl = [t.lower() for t in self.config.whitelist_terms]
            mask = candidates["rule_text"].str.lower().apply(lambda t: any(w in t for w in wl))
            if not any(w in query.lower() for w in wl):
                # If query itself doesn't match whitelist, keep mask only
                candidates = candidates[mask]
        # Blacklist: remove any rule containing any blacklisted term
        if self.config.blacklist_terms:
            bl = [t.lower() for t in self.config.blacklist_terms]
            mask = ~candidates["rule_text"].str.lower().apply(lambda t: any(b in t for b in bl))
            candidates = candidates[mask]
        return candidates

    def get_rule_text(self, rule_num: str) -> Optional[str]:
        return self.rule_num_to_text.get(rule_num)

    def _bm25_scores(self, query: str) -> Optional[np.ndarray]:
        if self._bm25 is None:
            return None
        tokens = self._tokenize(query)
        scores = np.array(self._bm25.get_scores(tokens))
        return scores

    def _dense_scores(self, query: str) -> Optional[np.ndarray]:
        if self._dense_model is None or self._dense_embeddings is None:
            return None
        q_emb = self._dense_model.encode([query], convert_to_tensor=True)
        sims = util.cos_sim(q_emb, self._dense_embeddings)[0].detach().cpu().numpy()
        return sims

    def _hybrid_rank(self, query: str) -> List[int]:
        bm25 = self._bm25_scores(query)
        dense = self._dense_scores(query)
        if bm25 is None and dense is None:
            # Fallback: keyword filter then return indices
            kws = set(self._tokenize(query))
            hits = [i for i, txt in enumerate(self.rules_df["rule_text"].tolist()) if kws & set(self._tokenize(txt))]
            return hits[: self.config.top_k]

        # Normalize each component
        def norm(x: np.ndarray) -> np.ndarray:
            x = x.astype(float)
            if np.allclose(x.max(), x.min()):
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())

        score = np.zeros(len(self.rules_df))
        if bm25 is not None:
            score += self.config.bm25_weight * norm(bm25)
        if dense is not None:
            score += self.config.dense_weight * norm(dense)

        top_idx = np.argsort(-score)[: self.config.top_k]
        return top_idx.tolist()

    def find_relevant_rules(self, term: str) -> List[str]:
        # Expand compound term separated by '/'
        subterms = [t.strip() for t in term.split('/') if t.strip()]
        if not subterms:
            subterms = [term]

        # Gather candidates from hybrid search
        candidates: List[Tuple[str, str]] = []  # (rule_num, rule_text)
        for sub in subterms:
            hits = self._hybrid_rank(sub)
            for i in hits:
                candidates.append((self.rules_df.iloc[i]["rule_num"], self.rules_df.iloc[i]["rule_text"]))

        # Deduplicate while preserving order
        seen = set()
        unique_candidates: List[Tuple[str, str]] = []
        for rn, rt in candidates:
            if rn not in seen:
                seen.add(rn)
                unique_candidates.append((rn, rt))

        # Apply WL/BL filters
        cand_df = pd.DataFrame(unique_candidates, columns=["rule_num", "rule_text"])
        cand_df = self._apply_filters(cand_df, term)

        # Include subrules (children) and referenced rules
        expanded = set(cand_df["rule_num"].tolist())
        # Children
        for rn in list(expanded):
            child_mask = self.rules_df["rule_num"].str.startswith(rn + ".")
            expanded.update(self.rules_df.loc[child_mask, "rule_num"].tolist())
        # Referenced rules in text
        for txt in cand_df["rule_text"].tolist():
            expanded.update(self._extract_rule_numbers(txt))

        # Return as sorted list with stable order based on first occurrence in corpus
        ordered = sorted(list(expanded), key=lambda r: self.rules_df.index[self.rules_df["rule_num"] == r].min() if any(self.rules_df["rule_num"] == r) else 1e9)
        return ordered

    def get_top_k_snippets(self, query: str, k: int = 2, char_limit: int = 200) -> List[str]:
        idxs = self._hybrid_rank(query)[:k]
        texts = [self.rules_df.iloc[i]["rule_text"] for i in idxs]
        return [t[:char_limit] for t in texts]

    # Convenience for serialization during debugging
    def dump_index_stats(self) -> Dict[str, int]:
        return {
            "num_rules": len(self.rules_df),
            "bm25": int(self._bm25 is not None),
            "dense": int(self._dense_model is not None and self._dense_embeddings is not None),
        }
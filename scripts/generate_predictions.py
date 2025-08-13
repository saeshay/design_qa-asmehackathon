# scripts/generate_predictions.py
import os
import csv
import time
import argparse
import pandas as pd
from typing import List, Dict

# Use router backends
try:
    from eval.model_router import (
        parse_model_map,
        choose_backend_for_subset,
        openai_chat,
        claude_chat,
        mock_chat,
    )
except ModuleNotFoundError:
    from eval.model_router import (
        parse_model_map,
        choose_backend_for_subset,
        openai_chat,
        claude_chat,
        mock_chat,
    )

SUBSETS = ["retrieval","compilation","definition","presence","dimension","functional_performance"]

def choose_backend(subset: str, model_map: Dict[str,str]) -> str:
    backend = choose_backend_for_subset(subset, model_map)
    if backend in ("anthropic","claude"):
        return "anthropic"
    if backend in ("openai", "mock"):
        return backend
    # fallback
    return "openai"

def build_messages(subset: str, row: dict) -> List[dict]:
    q = str(row.get("question","")).strip()
    if subset == "retrieval":
        prompt = f"Question: {q}\nReturn a concise answer using only relevant terms."
    elif subset == "compilation":
        prompt = f"Question: {q}\nAssemble a rule-respecting answer. Be brief and specific."
    elif subset == "definition":
        mentions = row.get("mentions","")
        prompt = f"Question: {q}\nMentions: {mentions}\nProvide a short, precise definition or description."
    elif subset == "presence":
        mentions = row.get("mentions","")
        prompt = f"Question: {q}\nMentions: {mentions}\nAnswer 'yes' or 'no' with one word if possible."
    elif subset == "dimension":
        dim_type = row.get("dimension_type","")
        # Keep it very constrained to help accuracy scoring
        prompt = (
            f"Question: {q}\nDimension Type: {dim_type}\n"
            "Return only the dimension result or label required; avoid extra text."
        )
    elif subset == "functional_performance":
        expl = row.get("explanation","")
        prompt = f"Question: {q}\nContext: {expl}\nAnswer concisely with the best functional performance outcome."
    else:
        prompt = f"Question: {q}\nAnswer briefly."

    sys = "You are a careful assistant. Follow instructions and be concise."
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": prompt},
    ]

def call_backend(backend: str, messages: List[dict]) -> str:
    # Light retry for transient failures
    for attempt in range(3):
        try:
            if backend == "openai":
                return openai_chat(messages, temperature=0.0, max_tokens=512).strip()
            elif backend == "anthropic":
                return claude_chat(messages, temperature=0.0, max_tokens=512).strip()
            elif backend == "mock":
                return mock_chat(messages).strip()
            else:
                return openai_chat(messages, temperature=0.0, max_tokens=512).strip()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(0.6 * (attempt + 1))
    return ""

def infer_model_map_from_env() -> Dict[str,str]:
    default = (os.getenv("DQ_PROVIDER") or "openai").strip().lower()
    entries = [f"default={default}"]
    for s in SUBSETS:
        v = os.getenv(f"DQ_MODEL_{s.upper()}")
        if v:
            entries.append(f"{s}={v.strip().lower()}")
    return parse_model_map(";".join(entries))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, choices=SUBSETS)
    ap.add_argument("--input", required=True, help="dataset CSV (may be *_who.csv etc.)")
    ap.add_argument("--output", required=True, help="destination model-output CSV with model_prediction column")
    ap.add_argument("--limit", type=int, default=None, help="optional cap on number of rows")
    ap.add_argument("--model-map", type=str, default=None,
                    help="e.g. 'default=openai;dimension=anthropic'")
    args = ap.parse_args()

    # Build model map
    model_map = parse_model_map(args.model_map) if args.model_map else infer_model_map_from_env()
    backend = choose_backend(args.subset, model_map)
    print(f"[INFO] Generating predictions for subset='{args.subset}' using backend='{backend}'")
    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)

    # Ensure we carry forward all original columns; add model_prediction
    rows = []
    for _, row in df.iterrows():
        messages = build_messages(args.subset, row.to_dict())
        pred = call_backend(backend, messages)
        out = row.to_dict()
        out["model_prediction"] = pred
        rows.append(out)

    out_df = pd.DataFrame(rows)
    # Persist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[INFO] Wrote {len(out_df)} rows -> {args.output}")

if __name__ == "__main__":
    main()
import argparse
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import shutil, time

from .rule_retriever import RuleAwareRetriever, RetrieverConfig
from .task_heads import (
    RetrievalHead,
    CompilationHead,
    DefinitionHead,
    PresenceHead,
    DimensionHead,
    FunctionalHead,
    EscalationBudget,
    answer_presence,
    answer_definition,
    answer_compilation,
    answer_retrieval,
    answer_dimension,
    answer_functional,
)
from .vlm_clients import MockClient, OpenAIClient, AnthropicClient
from .validators import limit_ocr


@dataclass
class Paths:
    # Datasets
    retrieval_csv: str = "dataset/rule_extraction/rule_retrieval_qa.csv"
    compilation_csv: str = "dataset/rule_extraction/rule_compilation_qa.csv"
    definition_csv: str = "dataset/rule_comprehension/rule_definition_qa.csv"
    presence_csv: str = "dataset/rule_comprehension/rule_presence_qa.csv"
    dimension_context_csv: str = "dataset/rule_compliance/rule_dimension_qa/context/rule_dimension_qa_context.csv"
    dimension_detailed_csv: str = "dataset/rule_compliance/rule_dimension_qa/detailed_context/rule_dimension_qa_detailed_context.csv"
    functional_csv: str = "dataset/rule_compliance/rule_functional_performance_qa/rule_functional_performance_qa.csv"
    # Images
    definition_images_dir: str = "dataset/rule_comprehension/rule_definition_qa"
    presence_images_dir: str = "dataset/rule_comprehension/rule_presence_qa"
    dimension_context_images_dir: str = "dataset/rule_compliance/rule_dimension_qa/context"
    dimension_detailed_images_dir: str = "dataset/rule_compliance/rule_dimension_qa/detailed_context"
    functional_images_dir: str = "dataset/rule_compliance/rule_functional_performance_qa/images"
    # Outputs
    out_dir: str = "your_outputs"


SYSTEM_PROMPTS = {
    "retrieval": "You are a precise rule retriever. Return exactly the rule text with no extra words.",
    "compilation": "You collect all relevant rule numbers. Return only rule numbers separated by commas.",
    "definition": "Identify the CAD component highlighted in pink. Return a short noun phrase only.",
    "presence": "Decide if the requested component is visible. Return yes or no only.",
    "dimension": "Read the drawing and answer: include 'Explanation:' then 'Answer: yes/no'.",
    "functional": "Use the image and rule to judge compliance. Include 'Explanation:' then 'Answer: yes/no'.",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_client(provider: str):
    provider = (provider or "mock").lower()
    if provider == "openai":
        return OpenAIClient(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if provider == "anthropic":
        return AnthropicClient(os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"))
    return MockClient()


def get_clients():
    provider = os.getenv("DQ_PROVIDER", "mock").lower()

    if provider == "hybrid":
        # OpenAI default, Anthropic escalation
        default_client = OpenAIClient(model=os.getenv("DQ_OPENAI_MODEL", "gpt-4o-mini"))
        escalation = AnthropicClient(model=os.getenv("DQ_ANTHROPIC_BIG", "claude-3-5-sonnet-20240620"))
        return default_client, escalation

    default_client = MockClient()
    escalation = None
    if provider == "openai":
        default_client = OpenAIClient(model=os.getenv("DQ_OPENAI_MODEL", "gpt-4o-mini"))
        escalation = OpenAIClient(model=os.getenv("DQ_OPENAI_BIG", "gpt-4o"))
    elif provider == "anthropic":
        default_client = AnthropicClient(model=os.getenv("DQ_ANTHROPIC_MODEL", "claude-3-haiku-20240307"))
        escalation = AnthropicClient(model=os.getenv("DQ_ANTHROPIC_BIG", "claude-3-5-sonnet-20240620"))
    return default_client, escalation


def _subset_client_for(provider: str, subset: str, default_client):
    """Optionally override model per subset via env vars.
    Env vars: DQ_MODEL_DIMENSION, DQ_MODEL_DEFINITION, DQ_MODEL_PRESENCE
    """
    model_env_map = {
        "dimension": os.getenv("DQ_MODEL_DIMENSION"),
        "definition": os.getenv("DQ_MODEL_DEFINITION"),
        "presence": os.getenv("DQ_MODEL_PRESENCE"),
    }
    desired = model_env_map.get(subset)
    if not desired:
        return default_client
    provider = (provider or os.getenv("DQ_PROVIDER", "mock")).lower()
    if provider == "openai":
        return OpenAIClient(model=desired)
    if provider == "anthropic":
        return AnthropicClient(model=desired)
    return default_client


def stitch_dimension_csv():
    out_dir = "your_outputs"
    cand_a = os.path.join(out_dir, "dimension_context.csv")
    cand_b = os.path.join(out_dir, "dimension_detailed.csv")
    final = os.path.join(out_dir, "dimension.csv")
    if os.path.exists(cand_a) and os.path.exists(cand_b):
        newer = cand_a if os.path.getmtime(cand_a) > os.path.getmtime(cand_b) else cand_b
        shutil.copy(newer, final)
        print(f"[dimension] Chose file from: {newer}")
    elif os.path.exists(cand_a):
        shutil.copy(cand_a, final)
        print(f"[dimension] Only found: {cand_a}")
    elif os.path.exists(cand_b):
        shutil.copy(cand_b, final)
        print(f"[dimension] Only found: {cand_b}")


def run_subset(subset: str, df: pd.DataFrame, default_client, escalation):
    budget = EscalationBudget(total_items=len(df))
    preds = []
    who = []
    # Per-subset client override
    provider = os.getenv("DQ_PROVIDER", "mock").lower()
    client_for_subset = _subset_client_for(provider, subset, default_client)
    for _, row in df.iterrows():
        # Trim any OCR payload
        if "ocr" in row and isinstance(row["ocr"], str):
            row["ocr"] = limit_ocr(row["ocr"], 200)
        if subset == "presence":
            p, w = answer_presence(row, client_for_subset, escalation, budget)
        elif subset == "definition":
            p, w = answer_definition(row, client_for_subset, escalation, budget)
        elif subset == "compilation":
            p, w = answer_compilation(row, client_for_subset, escalation, budget)
        elif subset == "retrieval":
            p, w = answer_retrieval(row, client_for_subset, escalation, budget)
        elif subset == "dimension":
            p, w = answer_dimension(row, client_for_subset, escalation, budget)
        elif subset == "functional":
            p, w = answer_functional(row, client_for_subset, escalation, budget)
        else:
            p, w = "INSUFFICIENT", "mini"
        preds.append(p)
        who.append(w)
        # Optional sleep to respect rate limits
        sleep_sec = float(os.getenv("DQ_SLEEP_SEC", "0"))
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    if subset == "dimension":
        stitch_dimension_csv()
    return preds, who, budget.used, budget.max_escalations


def run_all(paths: Paths, provider: Optional[str] = None):
    ensure_dir(paths.out_dir)

    # Select client via explicit provider override or env-driven get_clients
    if provider is not None:
        default_client = build_client(provider)
        escalation = None
    else:
        default_client, escalation = get_clients()

    retriever = RuleAwareRetriever(RetrieverConfig())

    # Rule extraction
    RetrievalHead(default_client, retriever, SYSTEM_PROMPTS["retrieval"], escalation=escalation).run(
        paths.retrieval_csv, os.path.join(paths.out_dir, "retrieval.csv")
    )
    CompilationHead(default_client, retriever, SYSTEM_PROMPTS["compilation"], escalation=escalation).run(
        paths.compilation_csv, os.path.join(paths.out_dir, "compilation.csv")
    )

    # Rule comprehension
    DefinitionHead(default_client, SYSTEM_PROMPTS["definition"], escalation=escalation).run(
        paths.definition_csv, paths.definition_images_dir, os.path.join(paths.out_dir, "definition.csv")
    )
    PresenceHead(default_client, SYSTEM_PROMPTS["presence"], escalation=escalation).run(
        paths.presence_csv, paths.presence_images_dir, os.path.join(paths.out_dir, "presence.csv")
    )

    # Rule compliance
    for subset_name, dim_csv, dim_dir in [
        ("dimension_context", paths.dimension_context_csv, paths.dimension_context_images_dir),
        ("dimension_detailed", paths.dimension_detailed_csv, paths.dimension_detailed_images_dir),
    ]:
        out_path = os.path.join(paths.out_dir, f"{subset_name}.csv")
        DimensionHead(default_client, SYSTEM_PROMPTS["dimension"], escalation=escalation).run(dim_csv, dim_dir, out_path)

    FunctionalHead(default_client, SYSTEM_PROMPTS["functional"], escalation=escalation).run(
        paths.functional_csv, paths.functional_images_dir, os.path.join(paths.out_dir, "functional_performance.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the full pipeline")
    parser.add_argument("--provider", choices=["mock", "openai", "anthropic"], default=None)
    args = parser.parse_args()
    if args.run:
        run_all(Paths(), provider=args.provider)
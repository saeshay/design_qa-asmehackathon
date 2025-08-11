import argparse
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .rule_retriever import RuleAwareRetriever, RetrieverConfig
from .task_heads import (
    RetrievalHead,
    CompilationHead,
    DefinitionHead,
    PresenceHead,
    DimensionHead,
    FunctionalHead,
)
from .vlm_clients import MockClient, OpenAIClient, AnthropicClient


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
    default_client = MockClient()
    escalation = None
    if provider == "openai":
        default_client = OpenAIClient(model=os.getenv("DQ_OPENAI_MODEL", "gpt-4o-mini"))
        escalation = OpenAIClient(model=os.getenv("DQ_OPENAI_BIG", "gpt-4o"))
    elif provider == "anthropic":
        default_client = AnthropicClient(model=os.getenv("DQ_ANTHROPIC_MODEL", "claude-3-haiku-20240307"))
        escalation = AnthropicClient(model=os.getenv("DQ_ANTHROPIC_BIG", "claude-3-5-sonnet-20240620"))
    return default_client, escalation


def run_all(paths: Paths, provider: Optional[str] = None):
    ensure_dir(paths.out_dir)

    # Select client via explicit provider override or env-driven get_clients
    if provider is not None:
        client = build_client(provider)
    else:
        client, _ = get_clients()

    retriever = RuleAwareRetriever(RetrieverConfig())

    # Rule extraction
    RetrievalHead(client, retriever, SYSTEM_PROMPTS["retrieval"]).run(
        paths.retrieval_csv, os.path.join(paths.out_dir, "retrieval.csv")
    )
    CompilationHead(client, retriever, SYSTEM_PROMPTS["compilation"]).run(
        paths.compilation_csv, os.path.join(paths.out_dir, "compilation.csv")
    )

    # Rule comprehension
    DefinitionHead(client, SYSTEM_PROMPTS["definition"]).run(
        paths.definition_csv, paths.definition_images_dir, os.path.join(paths.out_dir, "definition.csv")
    )
    PresenceHead(client, SYSTEM_PROMPTS["presence"]).run(
        paths.presence_csv, paths.presence_images_dir, os.path.join(paths.out_dir, "presence.csv")
    )

    # Rule compliance
    for subset_name, dim_csv, dim_dir in [
        ("dimension_context", paths.dimension_context_csv, paths.dimension_context_images_dir),
        ("dimension_detailed", paths.dimension_detailed_csv, paths.dimension_detailed_images_dir),
    ]:
        out_path = os.path.join(paths.out_dir, f"{subset_name}.csv")
        DimensionHead(client, SYSTEM_PROMPTS["dimension"]).run(dim_csv, dim_dir, out_path)

    FunctionalHead(client, SYSTEM_PROMPTS["functional"]).run(
        paths.functional_csv, paths.functional_images_dir, os.path.join(paths.out_dir, "functional_performance.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the full pipeline")
    parser.add_argument("--provider", choices=["mock", "openai", "anthropic"], default=None)
    args = parser.parse_args()
    if args.run:
        run_all(Paths(), provider=args.provider)
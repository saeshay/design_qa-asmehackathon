import argparse
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .rule_retriever import RuleAwareRetriever, RetrieverConfig
from .vlm_clients import MockClient, Ensemble
from .task_heads import (
    RetrievalHead,
    CompilationHead,
    DefinitionHead,
    PresenceHead,
    DimensionHead,
    FunctionalHead,
)


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


def run_all(paths: Paths):
    ensure_dir(paths.out_dir)

    # Clients and ensemble: plug in real clients here (Claude/GPT-4o) and set confidences
    clients = [MockClient("vlm_a"), MockClient("vlm_b")]
    ensemble = Ensemble(clients)

    retriever = RuleAwareRetriever(RetrieverConfig())

    # Rule extraction
    RetrievalHead(ensemble, retriever, SYSTEM_PROMPTS["retrieval"]).run(
        paths.retrieval_csv, os.path.join(paths.out_dir, "retrieval.csv")
    )
    CompilationHead(ensemble, retriever, SYSTEM_PROMPTS["compilation"]).run(
        paths.compilation_csv, os.path.join(paths.out_dir, "compilation.csv")
    )

    # Rule comprehension
    DefinitionHead(ensemble, SYSTEM_PROMPTS["definition"]).run(
        paths.definition_csv, paths.definition_images_dir, os.path.join(paths.out_dir, "definition.csv")
    )
    PresenceHead(ensemble, SYSTEM_PROMPTS["presence"]).run(
        paths.presence_csv, paths.presence_images_dir, os.path.join(paths.out_dir, "presence.csv")
    )

    # Rule compliance (two dimension subsets averaged by evaluator; we will concat into one CSV for each subset run)
    for subset_name, dim_csv, dim_dir in [
        ("dimension_context", paths.dimension_context_csv, paths.dimension_context_images_dir),
        ("dimension_detailed", paths.dimension_detailed_csv, paths.dimension_detailed_images_dir),
    ]:
        out_path = os.path.join(paths.out_dir, f"{subset_name}.csv")
        DimensionHead(ensemble, SYSTEM_PROMPTS["dimension"]).run(dim_csv, dim_dir, out_path)

    FunctionalHead(ensemble, SYSTEM_PROMPTS["functional"]).run(
        paths.functional_csv, paths.functional_images_dir, os.path.join(paths.out_dir, "functional_performance.csv")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the full pipeline")
    args = parser.parse_args()
    if args.run:
        run_all(Paths())
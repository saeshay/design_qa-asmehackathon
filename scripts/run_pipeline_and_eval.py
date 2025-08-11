import os
import sys
import subprocess

from pathlib import Path
import argparse

from scripts.pipeline.orchestrate import run_all, Paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["retrieval","compilation","definition","presence","dimension","functional","all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    paths = Paths()

    # Run pipeline
    from scripts.pipeline.orchestrate import get_clients
    default_client, escalation = get_clients()

    # Ensure outputs dir
    Path(paths.out_dir).mkdir(exist_ok=True)

    # Dispatch by subset for cheap runs
    from scripts.pipeline.rule_retriever import RetrieverConfig, RuleAwareRetriever
    from scripts.pipeline.task_heads import (
        RetrievalHead, CompilationHead, DefinitionHead, PresenceHead, DimensionHead, FunctionalHead
    )

    retriever = RuleAwareRetriever(RetrieverConfig())

    if args.subset in ("retrieval", "all"):
        RetrievalHead(default_client, retriever, "You are a precise rule retriever. Return exactly the rule text with no extra words.", escalation=escalation).run(
            paths.retrieval_csv, os.path.join(paths.out_dir, "retrieval.csv"), limit=args.limit
        )
    if args.subset in ("compilation", "all"):
        CompilationHead(default_client, retriever, "You collect all relevant rule numbers. Return only rule numbers separated by commas.", escalation=escalation).run(
            paths.compilation_csv, os.path.join(paths.out_dir, "compilation.csv"), limit=args.limit
        )
    if args.subset in ("definition", "all"):
        DefinitionHead(default_client, "Identify the CAD component highlighted in pink. Return a short noun phrase only.", escalation=escalation).run(
            paths.definition_csv, paths.definition_images_dir, os.path.join(paths.out_dir, "definition.csv"), limit=args.limit
        )
    if args.subset in ("presence", "all"):
        PresenceHead(default_client, "Decide if the requested component is visible. Return yes or no only.", escalation=escalation).run(
            paths.presence_csv, paths.presence_images_dir, os.path.join(paths.out_dir, "presence.csv"), limit=args.limit
        )
    if args.subset in ("dimension", "all"):
        # run both and concat
        DimensionHead(default_client, "Read the drawing and answer: include 'Explanation:' then 'Answer: yes/no'.", escalation=escalation).run(
            paths.dimension_context_csv, paths.dimension_context_images_dir, os.path.join(paths.out_dir, "dimension_context.csv"), limit=args.limit
        )
        DimensionHead(default_client, "Read the drawing and answer: include 'Explanation:' then 'Answer: yes/no'.", escalation=escalation).run(
            paths.dimension_detailed_csv, paths.dimension_detailed_images_dir, os.path.join(paths.out_dir, "dimension_detailed.csv"), limit=args.limit
        )
        # stitch
        ctx = Path("your_outputs/dimension_context.csv")
        det = Path("your_outputs/dimension_detailed.csv")
        if ctx.exists() and det.exists():
            import pandas as pd
            pd.concat([pd.read_csv(ctx), pd.read_csv(det)], ignore_index=True).to_csv("your_outputs/dimension.csv", index=False)
    if args.subset in ("functional", "all"):
        FunctionalHead(default_client, "Use the image and rule to judge compliance. Include 'Explanation:' then 'Answer: yes/no'.", escalation=escalation).run(
            paths.functional_csv, paths.functional_images_dir, os.path.join(paths.out_dir, "functional_performance.csv"), limit=args.limit
        )

    # Now run full evaluation if subset is all
    if args.subset == "all":
        cmd = [
            sys.executable, "eval/full_evaluation.py",
            "--path_to_retrieval", "your_outputs/retrieval.csv",
            "--path_to_compilation", "your_outputs/compilation.csv",
            "--path_to_definition", "your_outputs/definition.csv",
            "--path_to_presence", "your_outputs/presence.csv",
            "--path_to_dimension", "your_outputs/dimension.csv",
            "--path_to_functional_performance", "your_outputs/functional_performance.csv",
            "--save_path", "results.txt",
        ]
        if Path("results.txt").exists():
            Path("results.txt").unlink()
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
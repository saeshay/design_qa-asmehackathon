import os
import sys
import subprocess

from pathlib import Path

from scripts.pipeline.orchestrate import run_all, Paths


def main():
    # Run pipeline
    run_all(Paths())


    # Build dimension aggregate CSV expected by eval: evaluator takes a single path for dimension
    # We will concatenate context and detailed_context CSVs
    ctx = Path("your_outputs/dimension_context.csv")
    det = Path("your_outputs/dimension_detailed.csv")
    dim_out = Path("your_outputs/dimension.csv")
    if ctx.exists() and det.exists():
        import pandas as pd
        pd.concat([pd.read_csv(ctx), pd.read_csv(det)], ignore_index=True).to_csv(dim_out, index=False)

    # Now run full evaluation
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
    # The full_evaluation prompts for overwrite. Force yes by removing existing file first.
    if Path("results.txt").exists():
        Path("results.txt").unlink()
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
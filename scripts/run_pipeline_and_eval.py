import os
import sys
import subprocess

from pathlib import Path
import argparse
import pandas as pd

from scripts.pipeline.orchestrate import run_all, Paths, get_clients, run_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["retrieval","compilation","definition","presence","dimension","functional","all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Debug provider/model info
    print(f"[INFO] DQ_PROVIDER: {os.getenv('DQ_PROVIDER')}")
    print(f"[INFO] DQ_OPENAI_MODEL: {os.getenv('DQ_OPENAI_MODEL')}")
    print(f"[INFO] DQ_ANTHROPIC_MODEL: {os.getenv('DQ_ANTHROPIC_MODEL')}")

    paths = Paths()

    default_client, escalation = get_clients()

    Path(paths.out_dir).mkdir(exist_ok=True)

    total_used = 0
    total_cap = 0

    if args.subset in ("retrieval", "all"):
        df = pd.read_csv(paths.retrieval_csv)
        if args.limit: df = df.head(args.limit)
        preds, who, used, cap = run_subset("retrieval", df.copy(), default_client, escalation)
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(os.path.join(paths.out_dir, "retrieval.csv"), index=False)
        pd.DataFrame({"who": who}).to_csv(os.path.join(paths.out_dir, "retrieval_who.csv"), index=False)
        print(f"[retrieval] escalations: {used}/{cap} | who: mini={who.count('mini')}, escalation={who.count('escalation')}")
        total_used += used; total_cap += cap

    if args.subset in ("compilation", "all"):
        df = pd.read_csv(paths.compilation_csv)
        if args.limit: df = df.head(args.limit)
        preds, who, used, cap = run_subset("compilation", df.copy(), default_client, escalation)
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(os.path.join(paths.out_dir, "compilation.csv"), index=False)
        pd.DataFrame({"who": who}).to_csv(os.path.join(paths.out_dir, "compilation_who.csv"), index=False)
        print(f"[compilation] escalations: {used}/{cap} | who: mini={who.count('mini')}, escalation={who.count('escalation')}")
        total_used += used; total_cap += cap

    if args.subset in ("definition", "all"):
        df = pd.read_csv(paths.definition_csv)
        if args.limit: df = df.head(args.limit)
        df = df.copy()
        df["image_path"] = df["image"].apply(lambda n: os.path.join(paths.definition_images_dir, str(n)))
        preds, who, used, cap = run_subset("definition", df.copy(), default_client, escalation)
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(os.path.join(paths.out_dir, "definition.csv"), index=False)
        pd.DataFrame({"who": who}).to_csv(os.path.join(paths.out_dir, "definition_who.csv"), index=False)
        print(f"[definition] escalations: {used}/{cap} | who: mini={who.count('mini')}, escalation={who.count('escalation')}")
        total_used += used; total_cap += cap

    if args.subset in ("presence", "all"):
        df = pd.read_csv(paths.presence_csv)
        if args.limit: df = df.head(args.limit)
        df = df.copy()
        df["image_path"] = df["image"].apply(lambda n: os.path.join(paths.presence_images_dir, str(n)))
        preds, who, used, cap = run_subset("presence", df.copy(), default_client, escalation)
        df_out = df.copy()
        df_out["model_prediction"] = preds
        df_out.to_csv(os.path.join(paths.out_dir, "presence.csv"), index=False)
        pd.DataFrame({"who": who}).to_csv(os.path.join(paths.out_dir, "presence_who.csv"), index=False)
        print(f"[presence] escalations: {used}/{cap} | who: mini={who.count('mini')}, escalation={who.count('escalation')}")
        total_used += used; total_cap += cap

    if args.subset in ("dimension", "all"):
        # context
        df_ctx = pd.read_csv(paths.dimension_context_csv)
        if args.limit: df_ctx = df_ctx.head(args.limit)
        df_ctx = df_ctx.copy()
        df_ctx["image_path"] = df_ctx["image"].apply(lambda n: os.path.join(paths.dimension_context_images_dir, str(n)))
        preds_ctx, who_ctx, used_ctx, cap_ctx = run_subset("dimension", df_ctx.copy(), default_client, escalation)
        out_ctx = df_ctx.copy(); out_ctx["model_prediction"] = preds_ctx
        out_ctx.to_csv(os.path.join(paths.out_dir, "dimension_context.csv"), index=False)
        # detailed
        df_det = pd.read_csv(paths.dimension_detailed_csv)
        if args.limit: df_det = df_det.head(args.limit)
        df_det = df_det.copy()
        df_det["image_path"] = df_det["image"].apply(lambda n: os.path.join(paths.dimension_detailed_images_dir, str(n)))
        preds_det, who_det, used_det, cap_det = run_subset("dimension", df_det.copy(), default_client, escalation)
        out_det = df_det.copy(); out_det["model_prediction"] = preds_det
        out_det.to_csv(os.path.join(paths.out_dir, "dimension_detailed.csv"), index=False)
        # stitch
        import pandas as _pd
        _pd.concat([out_ctx, out_det], ignore_index=True).to_csv(os.path.join(paths.out_dir, "dimension.csv"), index=False)
        print(f"[dimension] escalations: {used_ctx+used_det}/{cap_ctx+cap_det} | who: mini={(who_ctx+who_det).count('mini')}, escalation={(who_ctx+who_det).count('escalation')}")
        total_used += used_ctx + used_det; total_cap += cap_ctx + cap_det

    if args.subset in ("functional", "all"):
        df = pd.read_csv(paths.functional_csv)
        if args.limit: df = df.head(args.limit)
        df = df.copy()
        df["image_path"] = df["image"].apply(lambda n: os.path.join(paths.functional_images_dir, str(n)))
        preds, who, used, cap = run_subset("functional", df.copy(), default_client, escalation)
        df_out = df.copy(); df_out["model_prediction"] = preds
        df_out.to_csv(os.path.join(paths.out_dir, "functional_performance.csv"), index=False)
        pd.DataFrame({"who": who}).to_csv(os.path.join(paths.out_dir, "functional_who.csv"), index=False)
        print(f"[functional] escalations: {used}/{cap} | who: mini={who.count('mini')}, escalation={who.count('escalation')}")
        total_used += used; total_cap += cap

    if total_cap > 0:
        print(f"[total] escalations: {total_used}/{total_cap}")

    # Full evaluation only when all
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
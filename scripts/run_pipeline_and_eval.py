# scripts/run_pipeline_and_eval.py
import os
import sys
import glob
import argparse
import subprocess

SUBSETS = ["retrieval","compilation","definition","presence","dimension","functional_performance"]

def load_dotenv_if_present():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
        print("[INFO] Loaded .env (if present).")
    except Exception:
        pass

def latest_csv_matching(keyword: str) -> str | None:
    files = [p for p in glob.glob("your_outputs/*.csv") if keyword in os.path.basename(p).lower()]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def has_model_prediction(csv_path: str) -> bool:
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, nrows=1)
        return "model_prediction" in df.columns
    except Exception:
        return False

def dataset_guess_for(subset: str) -> str | None:
    """
    Prefer '*_who.csv' as dataset; else any CSV that matches subset AND lacks 'model_prediction'.
    """
    candidates = [p for p in glob.glob("your_outputs/*.csv") if subset in os.path.basename(p).lower()]
    if not candidates:
        return None
    who = [p for p in candidates if "who" in os.path.basename(p).lower()]
    if who:
        who.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return who[0]
    import pandas as pd
    for p in sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True):
        try:
            df = pd.read_csv(p, nrows=1)
            if "model_prediction" not in df.columns:
                return p
        except Exception:
            continue
    return None

def ensure_predictions(subset: str, model_map_arg: str | None, limit: int | None, force: bool = False):
    newest = latest_csv_matching(subset)
    if newest and has_model_prediction(newest) and not force:
        print(f"[INFO] {subset}: using existing predictions -> {newest}")
        return

    ds = dataset_guess_for(subset)
    if not ds:
        # Try to synthesize a _who.csv from an existing output file by stripping model_prediction
        out_csv = os.path.join("your_outputs", f"{subset}.csv")
        if os.path.exists(out_csv):
            try:
                import pandas as _pd
                _cols = _pd.read_csv(out_csv, nrows=1).columns.str.lower().tolist()
                if "model_prediction" in _cols:
                    print(f"[INFO] {subset}: synthesizing {subset}_who.csv from {out_csv}")
                    gen_who = [sys.executable, os.path.join("scripts","make_who_from_outputs.py"),
                               "--subset", subset, "--input", out_csv, "--outdir", "your_outputs", "--force"]
                    _env = os.environ.copy()
                    _env["PYTHONPATH"] = os.getcwd() + (os.pathsep + _env.get("PYTHONPATH",""))
                    import subprocess as _sp
                    _sp.check_call(gen_who, env=_env)
                    ds = os.path.join("your_outputs", f"{subset}_who.csv")
                else:
                    print(f"[WARN] {subset}: no dataset CSV found and {out_csv} has no model_prediction to strip.")
            except Exception as e:
                print(f"[WARN] {subset}: failed to synthesize _who from {out_csv}: {e}")
        if not ds:
            print(f"[WARN] {subset}: no dataset CSV found to generate predictions (looked in your_outputs/). Skipping generation.")
            return

    out = os.path.join("your_outputs", f"{subset}.csv")
    gen_path = os.path.join("scripts", "generate_predictions.py")
    cmd = [sys.executable, gen_path,
           "--subset", subset, "--input", ds, "--output", out]
    if limit:
        cmd += ["--limit", str(limit)]
    if model_map_arg:
        cmd += ["--model-map", model_map_arg]
    print("[INFO] Generating:", " ".join(cmd))

    # NEW: ensure the repo root is on PYTHONPATH for the child process
    repo_root = os.path.dirname(os.path.abspath(__file__))  # .../scripts -> take parent below
    repo_root = os.path.dirname(repo_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = (repo_root + os.pathsep + env.get("PYTHONPATH", "")) if "PYTHONPATH" in env else repo_root

    subprocess.check_call(cmd, env=env)

def normalize_subset_arg(s: str) -> str:
    s = s.strip().lower()
    if s == "functional":
        return "functional_performance"
    return s

def main():
    load_dotenv_if_present()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=["retrieval","compilation","definition","presence","dimension","functional","functional_performance","all"],
        default="all",
        help="Which subset(s) to target (default: all). 'functional' is an alias for 'functional_performance'."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per subset when generating.")
    parser.add_argument("--generate", action="store_true", help="Generate predictions before evaluation.")
    parser.add_argument(
        "--model-map",
        type=str,
        default=None,
        help="Route backends per subset, e.g. 'default=openai;dimension=anthropic;functional_performance=anthropic'."
    )
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration of predictions for the target subset(s).")

    # Allow unknown args to flow through to eval.full_evaluation (keep its CLI intact)
    args, unknown = parser.parse_known_args()

    # Resolve subset list
    if args.subset == "all":
        subsets = SUBSETS[:]
    else:
        subsets = [normalize_subset_arg(args.subset)]

    # Optional generation phase
    if args.generate:
        for s in subsets:
            ensure_predictions(s, args.model_map, args.limit, args.regenerate)

    # Always launch the evaluator afterward
    cmd = [sys.executable, "-m", "eval.full_evaluation", "--overwrite"] + unknown
    print("[INFO] Launching evaluator:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
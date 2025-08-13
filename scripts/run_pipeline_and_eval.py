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
    # Mimic evaluator's approach: pick newest *.csv matching subset keyword
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
    # Prefer *who.csv as dataset; else any matching without model_prediction
    candidates = [p for p in glob.glob("your_outputs/*.csv") if subset in os.path.basename(p).lower()]
    who = [p for p in candidates if "who" in os.path.basename(p).lower()]
    if who:
        who.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return who[0]
    # fallback: first candidate lacking model_prediction
    import pandas as pd
    for p in candidates:
        try:
            df = pd.read_csv(p, nrows=1)
            if "model_prediction" not in df.columns:
                return p
        except Exception:
            continue
    return None

def ensure_predictions(subset: str, model_map_arg: str | None, limit: int | None):
    # If newest matching CSV lacks model_prediction, try to generate to your_outputs/<subset>.csv
    newest = latest_csv_matching(subset)
    if newest and has_model_prediction(newest):
        print(f"[INFO] {subset}: using existing predictions -> {newest}")
        return

    ds = dataset_guess_for(subset)
    if not ds:
        print(f"[WARN] {subset}: no dataset CSV found to generate predictions (looked in your_outputs/). Skipping generation.")
        return

    out = os.path.join("your_outputs", f"{subset}.csv")
    cmd = [sys.executable, "-m", "scripts.generate_predictions",
           "--subset", subset, "--input", ds, "--output", out]
    if limit:
        cmd += ["--limit", str(limit)]
    if model_map_arg:
        cmd += ["--model-map", model_map_arg]
    print("[INFO] Generating:", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    load_dotenv_if_present()
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="Generate predictions for subsets lacking model_prediction before evaluation")
    ap.add_argument("--limit", type=int, default=None, help="Row cap for generation (for quick tests)")
    ap.add_argument("--model-map", type=str, default=None, help="Route backends per subset, e.g. 'default=openai;dimension=anthropic'")
    args, unknown = ap.parse_known_args()

    if args.generate:
        for s in SUBSETS:
            ensure_predictions(s, args.model_map, args.limit)

    # Always run the evaluator after (or even without) generation
    cmd = [sys.executable, "-m", "eval.full_evaluation"] + unknown
    print("[INFO] Launching evaluator:", " ".join(cmd))
    os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main()
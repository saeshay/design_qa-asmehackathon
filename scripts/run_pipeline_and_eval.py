# scripts/run_pipeline_and_eval.py
import os
import sys
import argparse

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
    except Exception:
        # dotenv is optional; continue if missing
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default=None,
                        help="Optional: evaluate a single subset (retrieval|compilation|definition|presence|dimension|functional_performance)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: limit number of rows for a quick test")
    args, unknown = parser.parse_known_args()

    # Build argv for eval.full_evaluation (keep CSV auto-detect intact)
    eval_mod = "eval.full_evaluation"
    eval_args = []

    # Execute as module, pass through unknown args unchanged
    cmd = [sys.executable, "-m", eval_mod] + eval_args + unknown
    print("[INFO] Launching:", " ".join(cmd))
    os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main()
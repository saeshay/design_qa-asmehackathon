# scripts/make_who_from_outputs.py
import os, sys, argparse
import pandas as pd

PRED_COLS = {"model_prediction", "prediction", "pred", "answer_text", "answer", "model_output"}

SUBSETS = [
    "retrieval",
    "compilation",
    "definition",
    "presence",
    "dimension",
    "functional_performance",
]

def guess_input(subset: str) -> str | None:
    # Prefer your_outputs/<subset>.csv
    p = os.path.join("your_outputs", f"{subset}.csv")
    return p if os.path.exists(p) else None

def make_who(input_csv: str, outdir: str, force: bool = False) -> str:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    # Drop only known prediction-like columns; keep everything else
    drop_cols = [c for c in df.columns if c.lower() in PRED_COLS]
    if not drop_cols:
        print(f"[WARN] No prediction columns found in {input_csv}. Writing a copy as _who.")
    else:
        print(f"[INFO] Dropping columns {drop_cols} from {input_csv}")
        df = df.drop(columns=drop_cols, errors="ignore")

    subset = os.path.splitext(os.path.basename(input_csv))[0]  # e.g., 'presence'
    out_name = f"{subset}_who.csv"
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, out_name)
    if os.path.exists(out_path) and not force:
        raise FileExistsError(f"Refusing to overwrite existing {out_path}. Use --force to replace.")
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} ({len(df)} rows)")
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Create * _who.csv dataset(s) from outputs by stripping prediction columns.")
    ap.add_argument("--subset", choices=SUBSETS + ["all"], required=True, help="Subset to convert, or 'all'.")
    ap.add_argument("--input", help="Optional explicit path to input CSV; defaults to your_outputs/<subset>.csv")
    ap.add_argument("--outdir", default="your_outputs", help="Directory to write *_who.csv (default: your_outputs)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing *_who.csv")
    args = ap.parse_args()

    subsets = SUBSETS if args.subset == "all" else [args.subset]
    for s in subsets:
        inp = args.input or guess_input(s)
        if not inp:
            print(f"[WARN] No input found for subset '{s}' (looked for your_outputs/{s}.csv). Skipping.")
            continue
        try:
            make_who(inp, args.outdir, force=args.force)
        except Exception as e:
            print(f"[ERROR] {s}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
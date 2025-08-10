# fix_eval_columns.py
import pandas as pd
from pathlib import Path

def yn(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    if s in {"yes","y","true","1"}: return "yes"
    if s in {"no","n","false","0"}: return "no"
    # try extract trailing "Answer: yes/no"
    if "answer:" in s:
        tail = s.split("answer:")[-1].strip().split()[0]
        if tail in {"yes","no"}: return tail
    return s  # fallback

def fix_file(p: Path):
    df = pd.read_csv(p)
    # Normalize the source column if present
    if "model_prediction" in df.columns:
        df["model_prediction"] = df["model_prediction"].map(yn)
    # Ensure evaluator-expected columns exist and are filled
    src = "model_prediction"
    for col in ["prediction", "model_prediction_answer"]:
        if col in df.columns or True:
            df[col] = df.get(src, "").map(yn)
    # (Optional) make explanations consistent too
    for col in ["explanation","model_explanation","prediction_explanation","model_prediction_explanation"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df.to_csv(p, index=False)
    mp = df["prediction"] if "prediction" in df.columns else df["model_prediction"]
    print(f"{p.name:28} rows: {len(df):4d} | empty preds: {((mp=='') | mp.isna()).sum()}")
    print("sample preds:", mp.head(2).tolist(), "\n")

base = Path("your_outputs")
for name in ["dimension","functional_performance"]:
    p = base / f"{name}.csv"
    if p.exists():
        fix_file(p)
    else:
        print(f"Missing: {p}")

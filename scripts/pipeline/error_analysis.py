import ast
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class FailureCase:
    idx: int
    question: str
    ground_truth: str
    prediction: str
    category: str  # retrieval_failure | reasoning_failure | format_error


def classify_retrieval_compilation(df: pd.DataFrame) -> List[FailureCase]:
    failures = []
    for i, row in df.iterrows():
        gt = str(row.get("ground_truth", ""))
        pred = str(row.get("model_prediction", ""))
        if pred == "INSUFFICIENT" or pred.strip() == "":
            failures.append(FailureCase(i, row["question"], gt, pred, "retrieval_failure"))
        else:
            failures.append(FailureCase(i, row["question"], gt, pred, "reasoning_failure"))
    return failures


def classify_yes_no(df: pd.DataFrame) -> List[FailureCase]:
    failures = []
    for i, row in df.iterrows():
        gt = str(row.get("ground_truth", ""))
        pred = str(row.get("model_prediction", ""))
        if pred == "INSUFFICIENT" or pred.strip() == "":
            failures.append(FailureCase(i, row["question"], gt, pred, "format_error"))
        else:
            failures.append(FailureCase(i, row["question"], gt, pred, "reasoning_failure"))
    return failures


def to_rerun_indices(failures: List[FailureCase], categories: Tuple[str, ...]) -> List[int]:
    cats = set(categories)
    return [f.idx for f in failures if f.category in cats]
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from config import (
    AGE_BUCKETS,
    AGE_BUCKET_LABELS,
    AGE_BUCKET_LABELS_EXCEL,
    AGE_BUCKET_MIDPOINTS,
    AGE_GROUPS,
    AGE_MAX,
    AGE_MIN,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_utkface_filename(filename: str) -> Optional[Dict]:
    """
    Expected formats like:
    26_0_2_20170104023102422.jpg.chip.jpg
    8_1_0_20170109203412345.jpg.chip.jpg
    """
    stem = Path(filename).name
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except ValueError:
        return None

    if age < AGE_MIN or age > AGE_MAX:
        return None
    if gender not in (0, 1):
        return None
    if race not in (0, 1, 2, 3, 4):
        return None

    return {"age": age, "gender": gender, "race": race}


def age_to_bucket_index(age: float) -> int:
    age = float(age)
    for idx, (low, high) in enumerate(AGE_BUCKETS):
        if low <= age <= high:
            return idx
    if age < AGE_BUCKETS[0][0]:
        return 0
    return len(AGE_BUCKETS) - 1


def age_to_bucket_label(age: float) -> str:
    return AGE_BUCKET_LABELS[age_to_bucket_index(age)]


def age_to_bucket_label_excel(age: float) -> str:
    return AGE_BUCKET_LABELS_EXCEL[age_to_bucket_index(age)]


def bucket_index_to_midpoint(bucket_idx: int) -> float:
    return AGE_BUCKET_MIDPOINTS[int(bucket_idx)]


def age_to_group(age: float) -> str:
    age = float(age)
    for group_name, (low, high) in AGE_GROUPS.items():
        if low <= age <= high:
            return group_name
    return "unknown"


def make_age_strata_labels(ages: pd.Series, num_bins: int = 12, min_count: int = 2) -> pd.Series:
    ages = pd.Series(ages).astype(float)
    try:
        strata = pd.qcut(ages, q=min(num_bins, ages.nunique()), duplicates="drop")
    except ValueError:
        strata = pd.cut(ages, bins=min(num_bins, max(2, ages.nunique())), include_lowest=True)

    strata = strata.astype(str)
    counts = strata.value_counts()
    rare = counts[counts < min_count].index
    if len(rare) > 0:
        bucket_fallback = ages.apply(age_to_bucket_label)
        strata = strata.where(~strata.isin(rare), bucket_fallback)
    return strata.astype(str)


def split_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    full_mean = float(df["age"].mean())
    full_std = float(df["age"].std(ddof=0))
    for split_name, sdf in df.groupby("split"):
        rows.append(
            {
                "split": split_name,
                "count": int(len(sdf)),
                "age_mean": float(sdf["age"].mean()),
                "age_std": float(sdf["age"].std(ddof=0)),
                "male_ratio": float((sdf["gender"] == 0).mean()),
                "female_ratio": float((sdf["gender"] == 1).mean()),
                "age_mean_delta_vs_all": float(abs(sdf["age"].mean() - full_mean)),
                "age_std_delta_vs_all": float(abs(sdf["age"].std(ddof=0) - full_std)),
            }
        )
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "age_mae": float(mae),
        "age_mse": float(mse),
        "age_rmse": float(rmse),
        "age_r2": float(r2),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = {
        "gender_accuracy": float(accuracy_score(y_true, y_pred)),
        "gender_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "gender_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "gender_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["gender_roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["gender_roc_auc"] = float("nan")
    return metrics


def compute_age_bucket_accuracy(y_true_age: np.ndarray, y_pred_age: np.ndarray) -> float:
    y_true_bucket = np.array([age_to_bucket_index(x) for x in y_true_age])
    y_pred_bucket = np.array([age_to_bucket_index(x) for x in y_pred_age])
    return float((y_true_bucket == y_pred_bucket).mean())


def compute_bucket_confusion(y_true_age: np.ndarray, y_pred_age: np.ndarray) -> np.ndarray:
    y_true_bucket = np.array([age_to_bucket_index(x) for x in y_true_age])
    y_pred_bucket = np.array([age_to_bucket_index(x) for x in y_pred_age])
    return confusion_matrix(y_true_bucket, y_pred_bucket, labels=list(range(len(AGE_BUCKETS))))


def build_group_metrics(df_pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for group_name in sorted(df_pred["age_group"].dropna().unique()):
        group_df = df_pred[df_pred["age_group"] == group_name].copy()
        if group_df.empty:
            continue
        row = {
            "group": group_name,
            "count": int(len(group_df)),
            "age_mae": float(mean_absolute_error(group_df["true_age"], group_df["pred_age"])),
            "gender_accuracy": float((group_df["true_gender"] == group_df["pred_gender"]).mean()),
        }
        rows.append(row)

    for gender_value in sorted(df_pred["true_gender"].dropna().unique()):
        gdf = df_pred[df_pred["true_gender"] == gender_value].copy()
        if gdf.empty:
            continue
        row = {
            "group": f"gender_{int(gender_value)}",
            "count": int(len(gdf)),
            "age_mae": float(mean_absolute_error(gdf["true_age"], gdf["pred_age"])),
            "gender_accuracy": float((gdf["true_gender"] == gdf["pred_gender"]).mean()),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def flatten_metrics_for_csv(metrics: Dict[str, float], method_name: str) -> pd.DataFrame:
    row = {"method": method_name}
    row.update(metrics)
    return pd.DataFrame([row])

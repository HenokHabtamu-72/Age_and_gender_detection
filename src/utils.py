import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from .config import AGE_MAX, AGE_MIN
except ImportError:
    from config import AGE_MAX, AGE_MIN


SPLIT_DISPLAY_ORDER = ['train', 'val', 'test']


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
    stem = Path(filename).name
    parts = stem.split('_')
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
    return {'age': age, 'gender': gender, 'race': race}



def _fallback_age_band_label(age: float) -> str:
    band_start = int(max(AGE_MIN, min(AGE_MAX, age)) // 10) * 10
    band_end = min(band_start + 9, AGE_MAX)
    return f'{band_start:02d}-{band_end:02d}'



def make_age_strata_labels(ages: pd.Series, num_bins: int = 12, min_count: int = 2) -> pd.Series:
    ages = pd.Series(ages).astype(float)
    try:
        strata = pd.qcut(ages, q=min(num_bins, ages.nunique()), duplicates='drop')
    except ValueError:
        strata = pd.cut(ages, bins=min(num_bins, max(2, ages.nunique())), include_lowest=True)
    strata = strata.astype(str)
    counts = strata.value_counts()
    rare = counts[counts < min_count].index
    if len(rare) > 0:
        fallback = ages.apply(_fallback_age_band_label)
        strata = strata.where(~strata.isin(rare), fallback)
    return strata.astype(str)



def choose_stratify_labels(df: pd.DataFrame) -> Optional[pd.Series]:
    candidate_builders = [
        lambda x: x['gender'].astype(str) + '_' + x['age_strata'].astype(str),
        lambda x: x['age_strata'].astype(str),
        lambda x: x['gender'].astype(str),
    ]
    for builder in candidate_builders:
        labels = builder(df)
        counts = labels.value_counts()
        if not counts.empty and int(counts.min()) >= 2:
            return labels
    return None



def _resolve_split_counts(n_samples: int, test_size: float | int) -> tuple[int, int]:
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError('test_size as a float must be between 0 and 1.')
        n_test = int(math.ceil(test_size * n_samples))
    else:
        n_test = int(test_size)
    n_train = n_samples - n_test
    return n_train, n_test



def _is_stratification_feasible(labels: pd.Series, n_samples: int, test_size: float | int) -> bool:
    if labels is None or labels.empty:
        return False
    counts = labels.value_counts()
    if counts.empty or int(counts.min()) < 2:
        return False
    n_train, n_test = _resolve_split_counts(n_samples=n_samples, test_size=test_size)
    if n_train <= 0 or n_test <= 0:
        return False
    n_classes = int(labels.nunique())
    return n_train >= n_classes and n_test >= n_classes



def stratified_split(df: pd.DataFrame, test_size: float, seed: int):
    stratify_labels = choose_stratify_labels(df)
    if _is_stratification_feasible(stratify_labels, n_samples=len(df), test_size=test_size):
        return train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify_labels)
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=None)



def split_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    full_mean = float(df['age'].mean())
    full_std = float(df['age'].std(ddof=0))
    for split_name, sdf in df.groupby('split'):
        rows.append({
            'split': split_name,
            'count': int(len(sdf)),
            'age_mean': float(sdf['age'].mean()),
            'age_std': float(sdf['age'].std(ddof=0)),
            'male_ratio': float((sdf['gender'] == 0).mean()),
            'female_ratio': float((sdf['gender'] == 1).mean()),
            'age_mean_delta_vs_all': float(abs(sdf['age'].mean() - full_mean)),
            'age_std_delta_vs_all': float(abs(sdf['age'].std(ddof=0) - full_std)),
        })
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    order_map = {name: idx for idx, name in enumerate(SPLIT_DISPLAY_ORDER)}
    summary_df['_split_order'] = summary_df['split'].map(lambda x: order_map.get(x, len(order_map)))
    return summary_df.sort_values(['_split_order', 'split']).drop(columns=['_split_order']).reset_index(drop=True)



def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value



def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe_value(data), f, indent=2, allow_nan=False)



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
    return {'age_mae': float(mae), 'age_mse': float(mse), 'age_rmse': float(rmse), 'age_r2': float(r2)}



def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        'gender_accuracy': float(accuracy_score(y_true, y_pred)),
        'gender_precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'gender_recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'gender_f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics['gender_roc_auc'] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics['gender_roc_auc'] = float('nan')
    return metrics



def flatten_metrics_for_csv(metrics: Dict[str, float], method_name: str, eval_split: Optional[str] = None) -> pd.DataFrame:
    row = {'method': method_name}
    if eval_split is not None:
        row['eval_split'] = eval_split
    row.update(metrics)
    return pd.DataFrame([row])

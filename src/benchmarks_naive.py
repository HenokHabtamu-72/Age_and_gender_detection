import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import PLOTS_DIR, RESULTS_DIR
from plots import plot_benchmark_bars
from utils import (
    age_to_group,
    build_group_metrics,
    classification_metrics,
    compute_age_bucket_accuracy,
    ensure_dir,
    flatten_metrics_for_csv,
    regression_metrics,
    save_dataframe,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="naive_baselines")
    args = parser.parse_args()

    split_df = pd.read_csv(args.split_csv)
    train_df = split_df[split_df["split"] == "train"].copy()
    test_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)

    age_mean = float(train_df["age"].mean())
    gender_majority = int(train_df["gender"].mode().iloc[0])
    gender_majority_prob = float(train_df["gender"].mean())

    pred_df = test_df.copy()
    pred_df["true_age"] = pred_df["age"]
    pred_df["true_gender"] = pred_df["gender"]
    pred_df["pred_age"] = age_mean
    pred_df["pred_gender"] = gender_majority
    pred_df["pred_gender_prob"] = gender_majority_prob
    pred_df["age_group"] = pred_df["true_age"].apply(age_to_group)

    y_true_age = pred_df["true_age"].to_numpy()
    y_pred_age = pred_df["pred_age"].to_numpy()
    y_true_gender = pred_df["true_gender"].to_numpy()
    y_pred_gender = pred_df["pred_gender"].to_numpy()
    y_prob_gender = pred_df["pred_gender_prob"].to_numpy()

    metrics = {}
    metrics.update(regression_metrics(y_true_age, y_pred_age))
    metrics.update(classification_metrics(y_true_gender, y_pred_gender, y_prob_gender))
    metrics["age_bucket_accuracy"] = compute_age_bucket_accuracy(y_true_age, y_pred_age)
    metrics["num_test_samples"] = int(len(pred_df))

    result_dir = RESULTS_DIR / args.experiment_name
    ensure_dir(result_dir)
    save_dataframe(pred_df, result_dir / "test_predictions.csv")
    save_json(metrics, result_dir / "metrics.json")
    flatten_metrics_for_csv(metrics, "naive_mean_age_majority_gender").to_csv(result_dir / "metrics_summary.csv", index=False)
    build_group_metrics(pred_df).to_csv(result_dir / "group_metrics.csv", index=False)

    benchmark_path = RESULTS_DIR / "benchmark_comparison.csv"
    rows = []
    if benchmark_path.exists():
        rows.append(pd.read_csv(benchmark_path))
    rows.append(flatten_metrics_for_csv(metrics, "naive_baseline"))
    benchmark_df = pd.concat(rows, ignore_index=True)
    benchmark_df = benchmark_df.drop_duplicates(subset=["method"], keep="last")
    benchmark_df.to_csv(benchmark_path, index=False)

    plot_benchmark_bars(benchmark_df, PLOTS_DIR / "benchmark_comparison")

    print(pd.DataFrame([metrics]).T)


if __name__ == "__main__":
    main()

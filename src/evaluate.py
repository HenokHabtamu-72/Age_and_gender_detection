import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import AGE_MAX, AGE_MIN, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, PLOTS_DIR, RESULTS_DIR
from dataset import UTKFaceDataset
from model import MultiTaskCNN
from plots import (
    plot_age_scatter,
    plot_bucket_confusion,
    plot_confusion,
    plot_group_metrics,
    plot_residual_hist,
    plot_residual_scatter,
    plot_roc_curve,
    plot_sample_predictions,
)
from utils import (
    age_to_bucket_index,
    age_to_group,
    build_group_metrics,
    classification_metrics,
    compute_age_bucket_accuracy,
    compute_bucket_confusion,
    ensure_dir,
    flatten_metrics_for_csv,
    regression_metrics,
    save_dataframe,
    save_json,
)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MultiTaskCNN(
        variant=ckpt["variant"],
        dropout=ckpt["dropout"],
        use_se=ckpt.get("use_se", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="improved_cnn")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    split_df = pd.read_csv(args.split_csv)
    test_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)

    if "age_bucket_idx" not in test_df.columns:
        test_df["age_bucket_idx"] = test_df["age"].apply(age_to_bucket_index)

    model, ckpt = load_model(Path(args.checkpoint), device)
    test_ds = UTKFaceDataset(test_df, image_size=ckpt["image_size"], train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            outputs = model(images)

            age_pred = outputs["age"].cpu().numpy()
            gender_prob = outputs["gender_prob"].cpu().numpy()
            gender_pred = (gender_prob >= 0.5).astype(int)
            bucket_prob = outputs["age_bucket_prob"].cpu().numpy()
            bucket_pred = bucket_prob.argmax(axis=1)

            for i in range(len(age_pred)):
                true_age = float(batch["age"][i].item())
                true_gender = int(batch["gender"][i].item())
                pred_age = float(np.clip(age_pred[i], AGE_MIN, AGE_MAX))
                pred_gender_prob = float(gender_prob[i])
                pred_gender = int(gender_pred[i])
                pred_bucket = int(bucket_pred[i])

                preds.append(
                    {
                        "image_path": batch["image_path"][i],
                        "filename": batch["filename"][i],
                        "true_age": true_age,
                        "pred_age": pred_age,
                        "true_gender": true_gender,
                        "pred_gender": pred_gender,
                        "pred_gender_prob": pred_gender_prob,
                        "true_age_bucket": age_to_bucket_index(true_age),
                        "pred_age_bucket": age_to_bucket_index(pred_age),
                        "pred_aux_age_bucket": pred_bucket,
                        "pred_aux_age_bucket_conf": float(bucket_prob[i, pred_bucket]),
                        "age_group": age_to_group(true_age),
                    }
                )

    pred_df = pd.DataFrame(preds)
    y_true_age = pred_df["true_age"].to_numpy()
    y_pred_age = pred_df["pred_age"].to_numpy()
    y_true_gender = pred_df["true_gender"].to_numpy()
    y_pred_gender = pred_df["pred_gender"].to_numpy()
    y_prob_gender = pred_df["pred_gender_prob"].to_numpy()

    metrics = {}
    metrics.update(regression_metrics(y_true_age, y_pred_age))
    metrics.update(classification_metrics(y_true_gender, y_pred_gender, y_prob_gender))
    metrics["age_bucket_accuracy"] = compute_age_bucket_accuracy(y_true_age, y_pred_age)
    metrics["aux_age_bucket_accuracy"] = float((pred_df["true_age_bucket"].to_numpy() == pred_df["pred_aux_age_bucket"].to_numpy()).mean())
    metrics["num_test_samples"] = int(len(pred_df))

    result_dir = RESULTS_DIR / args.experiment_name
    plot_dir = PLOTS_DIR / args.experiment_name
    ensure_dir(result_dir)
    ensure_dir(plot_dir)

    save_dataframe(pred_df, result_dir / "test_predictions.csv")
    save_json(metrics, result_dir / "metrics.json")
    flatten_metrics_for_csv(metrics, args.experiment_name).to_csv(result_dir / "metrics_summary.csv", index=False)

    benchmark_path = RESULTS_DIR / "benchmark_comparison.csv"
    rows = []
    if benchmark_path.exists():
        rows.append(pd.read_csv(benchmark_path))
    rows.append(flatten_metrics_for_csv(metrics, args.experiment_name))
    benchmark_df = pd.concat(rows, ignore_index=True)
    benchmark_df = benchmark_df.drop_duplicates(subset=["method"], keep="last")
    benchmark_df.to_csv(benchmark_path, index=False)

    group_df = build_group_metrics(pred_df)
    save_dataframe(group_df, result_dir / "group_metrics.csv")

    plot_age_scatter(y_true_age, y_pred_age, plot_dir / "age_true_vs_pred.png")
    plot_residual_hist(y_true_age, y_pred_age, plot_dir / "age_residual_hist.png")
    plot_residual_scatter(y_true_age, y_pred_age, plot_dir / "age_residual_scatter.png")
    plot_confusion(y_true_gender, y_pred_gender, plot_dir / "gender_confusion_matrix.png", labels=[0, 1], title="Gender confusion matrix")
    if len(np.unique(y_true_gender)) > 1:
        plot_roc_curve(y_true_gender, y_prob_gender, plot_dir / "gender_roc_curve.png")
    plot_bucket_confusion(compute_bucket_confusion(y_true_age, y_pred_age), plot_dir / "age_bucket_confusion_matrix.png")
    plot_group_metrics(group_df, plot_dir / "group_age_mae.png")
    plot_sample_predictions(pred_df, plot_dir / "sample_predictions.png")

    print("Evaluation finished.")
    print(pd.DataFrame([metrics]).T)


if __name__ == "__main__":
    main()

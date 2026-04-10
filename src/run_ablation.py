import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR


EXPERIMENTS = [
    {
        "experiment_name": "ablation_baseline",
        "variant": "baseline",
        "epochs": 20,
        "dropout": 0.30,
        "age_loss": "huber",
    },
    {
        "experiment_name": "ablation_improved",
        "variant": "improved",
        "epochs": 25,
        "dropout": 0.35,
        "age_loss": "huber",
    },
    {
        "experiment_name": "ablation_improved_mse",
        "variant": "improved",
        "epochs": 25,
        "dropout": 0.35,
        "age_loss": "mse",
    },
]


def run_command(command):
    print(" ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    rows = []
    for exp in EXPERIMENTS:
        train_cmd = [
            sys.executable,
            str(script_dir / "train.py"),
            "--split_csv",
            args.split_csv,
            "--experiment_name",
            exp["experiment_name"],
            "--variant",
            exp["variant"],
            "--epochs",
            str(exp["epochs"]),
            "--dropout",
            str(exp["dropout"]),
            "--age_loss",
            exp["age_loss"],
        ]
        run_command(train_cmd)

        eval_cmd = [
            sys.executable,
            str(script_dir / "evaluate.py"),
            "--split_csv",
            args.split_csv,
            "--checkpoint",
            str(Path("outputs/models") / exp["experiment_name"] / "best_model.pt"),
            "--experiment_name",
            exp["experiment_name"],
        ]
        run_command(eval_cmd)

        metrics_path = RESULTS_DIR / exp["experiment_name"] / "metrics_summary.csv"
        metrics_df = pd.read_csv(metrics_path)
        metrics_df["variant"] = exp["variant"]
        metrics_df["age_loss"] = exp["age_loss"]
        rows.append(metrics_df)

    ablation_df = pd.concat(rows, ignore_index=True)
    ablation_df.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)
    print(ablation_df)


if __name__ == "__main__":
    main()

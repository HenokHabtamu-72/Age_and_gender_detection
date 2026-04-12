import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    from .config import MODELS_DIR, RESULTS_DIR, resolve_project_path
except ImportError:
    from config import MODELS_DIR, RESULTS_DIR, resolve_project_path



EXPERIMENTS = [
    {
        "experiment_name": "ablation_baseline",
        "variant": "baseline",
        "epochs": 20,
        "dropout": 0.30,
        "age_loss": "huber",
    },
    {
        "experiment_name": "ablation_improved_with_se",
        "variant": "improved_with_se",
        "epochs": 25,
        "dropout": 0.35,
        "age_loss": "huber",
    },
    {
        "experiment_name": "ablation_improved_mse_with_se",
        "variant": "improved_with_se",
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
    parser.add_argument('--split_csv', type=str, required=True)
    parser.add_argument('--eval_split', type=str, default='test')
    args = parser.parse_args()
    split_csv = str(resolve_project_path(args.split_csv))
    script_dir = Path(__file__).resolve().parent
    rows = []
    for exp in EXPERIMENTS:
        run_command([sys.executable, str(script_dir / 'train.py'), '--split_csv', split_csv, '--experiment_name', exp['experiment_name'], '--variant', exp['variant'], '--epochs', str(exp['epochs']), '--dropout', str(exp['dropout']), '--age_loss', exp['age_loss']])
        run_command([sys.executable, str(script_dir / 'evaluate.py'), '--split_csv', split_csv, '--checkpoint', str(MODELS_DIR / exp['experiment_name'] / 'best_model.pt'), '--experiment_name', exp['experiment_name'], '--split_name', args.eval_split])
        metrics_df = pd.read_csv(RESULTS_DIR / exp['experiment_name'] / f'metrics_summary_{args.eval_split}.csv')
        metrics_df['variant'] = exp['variant']
        metrics_df['age_loss'] = exp['age_loss']
        rows.append(metrics_df)
    ablation_df = pd.concat(rows, ignore_index=True)
    ablation_df.to_csv(RESULTS_DIR / f'ablation_summary_{args.eval_split}.csv', index=False)
    print(ablation_df)

if __name__ == "__main__":
    main()

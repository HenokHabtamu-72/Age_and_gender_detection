import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import AGE_MAX, AGE_MIN, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, PLOTS_DIR, RESULTS_DIR, resolve_project_path
    from .dataset import UTKFaceDataset
    from .model import MultiTaskCNN
    from .plots import plot_age_scatter, plot_benchmark_bars, plot_confusion, plot_residual_hist, plot_residual_scatter, plot_roc_curve, plot_sample_predictions
    from .utils import classification_metrics, ensure_dir, flatten_metrics_for_csv, regression_metrics, save_dataframe, save_json
except ImportError:
    from config import AGE_MAX, AGE_MIN, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, PLOTS_DIR, RESULTS_DIR, resolve_project_path
    from dataset import UTKFaceDataset
    from model import MultiTaskCNN
    from plots import plot_age_scatter, plot_benchmark_bars, plot_confusion, plot_residual_hist, plot_residual_scatter, plot_roc_curve, plot_sample_predictions
    from utils import classification_metrics, ensure_dir, flatten_metrics_for_csv, regression_metrics, save_dataframe, save_json



def resolve_checkpoint_model_options(ckpt: dict) -> Tuple[str, bool]:
    variant = ckpt.get('variant', 'baseline')
    use_se = bool(ckpt.get('use_se', False))
    if variant == 'improved_with_se':
        return 'improved_with_se', True
    if variant == 'improved':
        return 'improved', use_se
    if variant == 'baseline':
        return 'baseline', use_se
    raise ValueError(f'Unsupported checkpoint variant: {variant}')



def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    variant, use_se = resolve_checkpoint_model_options(ckpt)
    model = MultiTaskCNN(variant=variant, dropout=ckpt['dropout'], use_se=use_se).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model, ckpt



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_csv', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--split_name', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    checkpoint_path = resolve_project_path(args.checkpoint)
    experiment_name = args.experiment_name or checkpoint_path.parent.name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    split_df = pd.read_csv(resolve_project_path(args.split_csv))
    eval_df = split_df[split_df['split'] == args.split_name].copy().reset_index(drop=True)
    if eval_df.empty:
        available_splits = sorted(split_df['split'].dropna().unique().tolist())
        raise ValueError(f"No rows found for split '{args.split_name}'. Available splits: {available_splits}")

    model, ckpt = load_model(checkpoint_path, device)
    eval_ds = UTKFaceDataset(eval_df, image_size=ckpt['image_size'], train=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    preds = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f'Evaluating {args.split_name}'):
            outputs = model(batch['image'].to(device))
            age_pred = outputs['age'].cpu().numpy()
            gender_prob = outputs['gender_prob'].cpu().numpy()
            gender_pred = (gender_prob >= 0.5).astype(int)
            for i in range(len(age_pred)):
                preds.append({
                    'image_path': batch['image_path'][i],
                    'filename': batch['filename'][i],
                    'true_age': float(batch['age'][i].item()),
                    'pred_age': float(np.clip(age_pred[i], AGE_MIN, AGE_MAX)),
                    'true_gender': int(batch['gender'][i].item()),
                    'pred_gender': int(gender_pred[i]),
                    'pred_gender_prob': float(gender_prob[i]),
                })

    pred_df = pd.DataFrame(preds)
    y_true_age = pred_df['true_age'].to_numpy()
    y_pred_age = pred_df['pred_age'].to_numpy()
    y_true_gender = pred_df['true_gender'].to_numpy()
    y_pred_gender = pred_df['pred_gender'].to_numpy()
    y_prob_gender = pred_df['pred_gender_prob'].to_numpy()

    metrics = {'eval_split': args.split_name}
    metrics.update(regression_metrics(y_true_age, y_pred_age))
    metrics.update(classification_metrics(y_true_gender, y_pred_gender, y_prob_gender))
    metrics[f'num_{args.split_name}_samples'] = int(len(pred_df))

    result_dir = RESULTS_DIR / experiment_name
    plot_dir = PLOTS_DIR / experiment_name
    ensure_dir(result_dir)
    ensure_dir(plot_dir)
    save_dataframe(pred_df, result_dir / f'{args.split_name}_predictions.csv')
    save_json(metrics, result_dir / f'metrics_{args.split_name}.json')
    flatten_metrics_for_csv(metrics, experiment_name, eval_split=args.split_name).to_csv(result_dir / f'metrics_summary_{args.split_name}.csv', index=False)

    benchmark_path = RESULTS_DIR / 'benchmark_comparison.csv'
    rows = [pd.read_csv(benchmark_path)] if benchmark_path.exists() else []
    rows.append(flatten_metrics_for_csv(metrics, experiment_name, eval_split=args.split_name))
    benchmark_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=['method', 'eval_split'], keep='last')
    benchmark_df.to_csv(benchmark_path, index=False)
    plot_benchmark_bars(benchmark_df[benchmark_df['eval_split'] == args.split_name], PLOTS_DIR / 'benchmark_comparison' / args.split_name)

    plot_age_scatter(y_true_age, y_pred_age, plot_dir / f'age_true_vs_pred_{args.split_name}.png')
    plot_residual_hist(y_true_age, y_pred_age, plot_dir / f'age_residual_hist_{args.split_name}.png')
    plot_residual_scatter(y_true_age, y_pred_age, plot_dir / f'age_residual_scatter_{args.split_name}.png')
    plot_confusion(y_true_gender, y_pred_gender, plot_dir / f'gender_confusion_matrix_{args.split_name}.png', labels=[0, 1], title=f'Gender confusion matrix ({args.split_name})')
    if len(np.unique(y_true_gender)) > 1:
        plot_roc_curve(y_true_gender, y_prob_gender, plot_dir / f'gender_roc_curve_{args.split_name}.png')
    plot_sample_predictions(pred_df, plot_dir / f'sample_predictions_{args.split_name}.png')
    print(f"Evaluation finished on split '{args.split_name}'.")
    print(pd.DataFrame([metrics]).T)


if __name__ == '__main__':
    main()

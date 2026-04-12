import argparse

import pandas as pd

try:
    from .config import PLOTS_DIR, RESULTS_DIR, resolve_project_path
    from .plots import plot_benchmark_bars
    from .utils import classification_metrics, ensure_dir, flatten_metrics_for_csv, regression_metrics, save_dataframe, save_json
except ImportError:
    from config import PLOTS_DIR, RESULTS_DIR, resolve_project_path
    from plots import plot_benchmark_bars
    from utils import classification_metrics, ensure_dir, flatten_metrics_for_csv, regression_metrics, save_dataframe, save_json



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_csv', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, default='naive_baseline')
    parser.add_argument('--split_name', type=str, default='test')
    args = parser.parse_args()

    split_df = pd.read_csv(resolve_project_path(args.split_csv))
    train_df = split_df[split_df['split'] == 'train'].copy()
    eval_df = split_df[split_df['split'] == args.split_name].copy()
    if train_df.empty or eval_df.empty:
        available_splits = sorted(split_df['split'].dropna().unique().tolist())
        raise ValueError(f"Required rows missing. Available splits: {available_splits}")

    mean_age = float(train_df['age'].mean())
    female_class_prior = float(train_df['gender'].mean())
    majority_gender = int(female_class_prior >= 0.5)

    pred_df = eval_df[['image_path', 'filename', 'age', 'gender']].copy()
    pred_df = pred_df.rename(columns={'age': 'true_age', 'gender': 'true_gender'})
    pred_df['pred_age'] = mean_age
    pred_df['pred_gender'] = majority_gender
    pred_df['pred_gender_prob'] = female_class_prior

    metrics = {'eval_split': args.split_name}
    metrics.update(regression_metrics(pred_df['true_age'].to_numpy(), pred_df['pred_age'].to_numpy()))
    metrics.update(classification_metrics(pred_df['true_gender'].to_numpy(), pred_df['pred_gender'].to_numpy(), pred_df['pred_gender_prob'].to_numpy()))
    metrics[f'num_{args.split_name}_samples'] = int(len(pred_df))
    result_dir = RESULTS_DIR / args.experiment_name
    ensure_dir(result_dir)
    save_dataframe(pred_df, result_dir / f'{args.split_name}_predictions.csv')
    save_json(metrics, result_dir / f'metrics_{args.split_name}.json')
    flatten_metrics_for_csv(metrics, args.experiment_name, eval_split=args.split_name).to_csv(result_dir / f'metrics_summary_{args.split_name}.csv', index=False)
    benchmark_path = RESULTS_DIR / 'benchmark_comparison.csv'
    rows = [pd.read_csv(benchmark_path)] if benchmark_path.exists() else []
    rows.append(flatten_metrics_for_csv(metrics, args.experiment_name, eval_split=args.split_name))
    benchmark_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=['method', 'eval_split'], keep='last')
    benchmark_df.to_csv(benchmark_path, index=False)
    plot_benchmark_bars(benchmark_df[benchmark_df['eval_split'] == args.split_name], PLOTS_DIR / 'benchmark_comparison' / args.split_name)
    print(f"Naive baseline evaluation finished on split '{args.split_name}'.")
    print(pd.DataFrame([metrics]).T)


if __name__ == '__main__':
    main()

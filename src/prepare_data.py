import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from .config import DEFAULT_SEED, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, PLOTS_DIR, RESULTS_DIR, resolve_project_path
    from .plots import plot_age_distribution, plot_gender_distribution, plot_split_age_boxplot, plot_split_distribution
    from .utils import ensure_dir, make_age_strata_labels, parse_utkface_filename, save_dataframe, save_json, set_seed, split_distribution_summary, stratified_split
except ImportError:
    from config import DEFAULT_SEED, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, PLOTS_DIR, RESULTS_DIR, resolve_project_path
    from plots import plot_age_distribution, plot_gender_distribution, plot_split_age_boxplot, plot_split_distribution
    from utils import ensure_dir, make_age_strata_labels, parse_utkface_filename, save_dataframe, save_json, set_seed, split_distribution_summary, stratified_split



def build_metadata(data_dir: Path) -> pd.DataFrame:
    records = []
    image_paths = sorted([p for p in data_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    for path in tqdm(image_paths, desc='Parsing UTKFace filenames'):
        parsed = parse_utkface_filename(path.name)
        if parsed is None:
            continue
        records.append({'image_path': str(path.resolve()), 'filename': path.name, 'age': parsed['age'], 'gender': parsed['gender'], 'race': parsed['race']})
    return pd.DataFrame(records)



def add_splits(df: pd.DataFrame, seed: int = DEFAULT_SEED, val_size: float = DEFAULT_VAL_SIZE, test_size: float = DEFAULT_TEST_SIZE) -> pd.DataFrame:
    df = df.copy()
    df['age_strata'] = make_age_strata_labels(df['age'], num_bins=12, min_count=2)
    holdout_size = val_size + test_size
    if holdout_size <= 0 or holdout_size >= 1:
        raise ValueError('val_size + test_size must be between 0 and 1.')

    train_df, temp_df = stratified_split(df, test_size=holdout_size, seed=seed)
    if temp_df.empty:
        raise ValueError('Holdout split is empty. Increase dataset size or reduce val_size/test_size.')

    temp_df = temp_df.copy()
    temp_df['age_strata'] = make_age_strata_labels(temp_df['age'], num_bins=8, min_count=2)
    relative_test_size = test_size / holdout_size
    if relative_test_size <= 0 or relative_test_size >= 1:
        raise ValueError('Derived test split ratio must be between 0 and 1.')

    val_df, test_df = stratified_split(temp_df, test_size=relative_test_size, seed=seed)
    if val_df.empty or test_df.empty:
        raise ValueError('Validation or test split is empty. Increase dataset size or reduce val_size/test_size.')

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    return pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True).drop(columns=['age_strata'], errors='ignore')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default=str(RESULTS_DIR / 'utkface_metadata.csv'))
    parser.add_argument('--split_csv', type=str, default=str(RESULTS_DIR / 'utkface_splits.csv'))
    parser.add_argument('--summary_csv', type=str, default=str(RESULTS_DIR / 'split_distribution_summary.csv'))
    parser.add_argument('--summary_json', type=str, default=str(RESULTS_DIR / 'split_distribution_summary.json'))
    parser.add_argument('--plot_dir', type=str, default=str(PLOTS_DIR / 'prepare_data'))
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    set_seed(args.seed)
    data_dir = resolve_project_path(args.data_dir)
    metadata_csv = resolve_project_path(args.output_csv)
    split_csv = resolve_project_path(args.split_csv)
    split_summary_path = resolve_project_path(args.summary_csv)
    split_summary_json = resolve_project_path(args.summary_json)
    prepare_plot_dir = resolve_project_path(args.plot_dir)
    ensure_dir(metadata_csv.parent)
    ensure_dir(split_csv.parent)
    ensure_dir(split_summary_path.parent)
    ensure_dir(split_summary_json.parent)
    ensure_dir(prepare_plot_dir)
    df = build_metadata(data_dir)
    if df.empty:
        raise ValueError('No valid UTKFace images were found. Check your folder path and filenames.')
    save_dataframe(df, metadata_csv)
    split_df = add_splits(df, seed=args.seed)
    save_dataframe(split_df, split_csv)
    split_summary_df = split_distribution_summary(split_df)
    save_dataframe(split_summary_df, split_summary_path)
    save_json(split_summary_df.to_dict(orient='records'), split_summary_json)
    plot_age_distribution(split_df, prepare_plot_dir / 'age_distribution.png')
    plot_gender_distribution(split_df, prepare_plot_dir / 'gender_distribution.png')
    plot_split_distribution(split_df, prepare_plot_dir / 'split_distribution.png')
    plot_split_age_boxplot(split_df, prepare_plot_dir / 'split_age_boxplot.png')
    print(f'Saved metadata CSV: {metadata_csv}')
    print(f'Saved split CSV:    {split_csv}')
    print(f'Saved split summary: {split_summary_path}')
    print(split_df['split'].value_counts())
    print(split_summary_df.to_string(index=False))


if __name__ == '__main__':
    main()

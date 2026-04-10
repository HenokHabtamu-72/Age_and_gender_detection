import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import DEFAULT_SEED
from plots import (
    plot_age_distribution,
    plot_gender_distribution,
    plot_split_distribution,
    plot_split_age_boxplot,
)
from utils import (
    age_to_bucket_label,
    age_to_bucket_label_excel,
    ensure_dir,
    parse_utkface_filename,
    save_dataframe,
    save_json,
    set_seed,
    split_distribution_summary,
)


def build_metadata(data_dir: Path) -> pd.DataFrame:
    records = []
    image_paths = sorted(
        [p for p in data_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    for path in tqdm(image_paths, desc="Parsing UTKFace filenames"):
        parsed = parse_utkface_filename(path.name)
        if parsed is None:
            continue

        record = {
            "image_path": str(path.resolve()),
            "filename": path.name,
            "age": parsed["age"],
            "gender": parsed["gender"],
            "race": parsed["race"],
            "age_bucket": age_to_bucket_label(parsed["age"]),
            "age_bucket_excel": age_to_bucket_label_excel(parsed["age"]),
        }
        records.append(record)

    return pd.DataFrame(records)


def add_splits(df: pd.DataFrame, seed: int = 42, test_size: float = 0.15, val_size: float = 0.15) -> pd.DataFrame:
    df = df.copy()

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        shuffle=True,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    return final_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="outputs/results/utkface_metadata.csv")
    parser.add_argument("--split_csv", type=str, default="outputs/results/utkface_splits.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    metadata_csv = Path(args.output_csv)
    split_csv = Path(args.split_csv)
    prepare_plot_dir = Path("outputs/plots/prepare_data")
    split_summary_path = Path("outputs/results/split_distribution_summary.csv")
    split_summary_json = Path("outputs/results/split_distribution_summary.json")

    ensure_dir(metadata_csv.parent)
    ensure_dir(split_csv.parent)
    ensure_dir(prepare_plot_dir)

    df = build_metadata(data_dir)
    if df.empty:
        raise ValueError("No valid UTKFace images were found. Check your folder path and filenames.")

    save_dataframe(df, metadata_csv)

    split_df = add_splits(df, seed=args.seed)
    save_dataframe(split_df, split_csv)

    split_summary_df = split_distribution_summary(split_df)
    save_dataframe(split_summary_df, split_summary_path)
    save_json(split_summary_df.to_dict(orient="records"), split_summary_json)

    plot_age_distribution(split_df, prepare_plot_dir / "age_distribution.png")
    plot_gender_distribution(split_df, prepare_plot_dir / "gender_distribution.png")
    plot_split_distribution(split_df, prepare_plot_dir / "split_distribution.png")
    plot_split_age_boxplot(split_df, prepare_plot_dir / "split_age_boxplot.png")

    print(f"Saved metadata CSV: {metadata_csv}")
    print(f"Saved split CSV:    {split_csv}")
    print(f"Saved split summary: {split_summary_path}")
    print(split_df["split"].value_counts())
    print(split_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
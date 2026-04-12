# UTKFace Age & Gender Prediction Project

A local Python project for **age regression** and **gender classification** on **UTKFace**.

## What it does

- Predicts **age as a regression target**
- Predicts **gender as a binary classification target**
- Trains a multitask CNN from scratch
- Saves **train / val / test** splits to CSV
- Exports training history, validation metrics, test metrics, plots, and sample predictions
- Includes a simple naive baseline runner and an ablation runner

## Available model variants

There are exactly two user-facing model options:

- `baseline`
- `improved_with_se`

`improved_with_se` uses the deeper improved backbone with squeeze-and-excitation enabled automatically.
`baseline` uses the smaller baseline backbone with SE disabled.

## Expected dataset filename format

```text
age_gender_race_timestamp.jpg.chip.jpg
```

Examples:

```text
26_0_2_20170104023102422.jpg.chip.jpg
8_1_0_20170109203412345.jpg.chip.jpg
```

## Suggested folder layout

```text
utk_age_gender_project/
  data/
    UTKFace/
      1_0_0_20170109150557335.jpg.chip.jpg
      2_1_2_20170116174525125.jpg.chip.jpg
      ...
  src/
  outputs/
    logs/
    models/
    plots/
    results/
```

## Project structure

```text
utk_age_gender_project/
  README.md
  requirements.txt
  src/
    __init__.py
    config.py
    utils.py
    prepare_data.py
    dataset.py
    model.py
    plots.py
    train.py
    evaluate.py
    inference.py
    benchmarks_naive.py
    run_ablation.py
    simple_transforms.py
  outputs/
    logs/
    models/
    plots/
    results/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build that matches your machine.

## Workflow

### 1) Prepare metadata and train / val / test splits

The split preparation step creates a held-out test split so final reporting can be done on unseen data. It first separates `train` from a temporary holdout set, then splits that holdout into `val` and `test` using the safest available stratification key among `gender + age_strata`, `age_strata`, `gender`, or no stratification if the dataset is too small.

```bash
python src/prepare_data.py --data_dir data/UTKFace --output_csv outputs/results/utkface_metadata.csv --split_csv outputs/results/utkface_splits.csv
```

### 2) Train a model

Training uses the `train` split and selects the best checkpoint using **validation total loss**. Early stopping and the LR scheduler also use that same validation criterion.

Baseline:

```bash
python src/train.py --split_csv outputs/results/utkface_splits.csv --experiment_name baseline_cnn --variant baseline --epochs 25 --batch_size 64 --image_size 128
```

Improved model with SE:

```bash
python src/train.py --split_csv outputs/results/utkface_splits.csv --experiment_name improved_with_se_cnn --variant improved_with_se --epochs 35 --batch_size 64 --image_size 128 --dropout 0.35 --age_loss huber
```

### 3) Evaluate a trained checkpoint on the held-out test split

`evaluate.py` now defaults to `test`, derives the experiment name from the checkpoint folder when not provided, and loads checkpoints strictly so architecture mismatches fail loudly.

```bash
python src/evaluate.py --split_csv outputs/results/utkface_splits.csv --checkpoint outputs/models/improved_with_se_cnn/best_model.pt --split_name test
```

### 4) Run the naive baseline on the held-out test split

```bash
python src/benchmarks_naive.py --split_csv outputs/results/utkface_splits.csv --split_name test
```

### 5) Run the ablation suite

```bash
python src/run_ablation.py --split_csv outputs/results/utkface_splits.csv --eval_split test
```

### 6) Run single-image inference

```bash
python src/inference.py --checkpoint outputs/models/improved_with_se_cnn/best_model.pt --image path/to/image.jpg
```

Inference prints only predicted age, predicted gender, and gender probability.

## Outputs

### Training outputs

- `outputs/models/<experiment_name>/best_model.pt`
- `outputs/models/<experiment_name>/train_config.json`
- `outputs/logs/<experiment_name>/history.csv`
- `outputs/plots/<experiment_name>/training_total_loss.png`
- `outputs/plots/<experiment_name>/training_age_mae.png`
- `outputs/plots/<experiment_name>/training_gender_accuracy.png`

### Evaluation outputs

- `outputs/results/<experiment_name>/<split_name>_predictions.csv`
- `outputs/results/<experiment_name>/metrics_<split_name>.json`
- `outputs/results/<experiment_name>/metrics_summary_<split_name>.csv`
- `outputs/plots/<experiment_name>/age_true_vs_pred_<split_name>.png`
- `outputs/plots/<experiment_name>/age_residual_hist_<split_name>.png`
- `outputs/plots/<experiment_name>/age_residual_scatter_<split_name>.png`
- `outputs/plots/<experiment_name>/gender_confusion_matrix_<split_name>.png`
- `outputs/plots/<experiment_name>/gender_roc_curve_<split_name>.png` when both gender classes are present
- `outputs/plots/<experiment_name>/sample_predictions_<split_name>.png`

### Optional benchmark comparison outputs

Running `benchmarks_naive.py` and `evaluate.py` will maintain split-aware comparison files:

- `outputs/results/benchmark_comparison.csv`
- `outputs/plots/benchmark_comparison/<split_name>/benchmark_bar_age_mae.png`
- `outputs/plots/benchmark_comparison/<split_name>/benchmark_bar_gender_accuracy.png`

## Notes

- Training and evaluation use GPU automatically when available.
- The project does not depend on `torchvision`.
- The workflow now keeps a held-out `test` split for final comparisons.
- Relative paths are resolved from the project root, so scripts behave consistently even when launched from another working directory.
- New training runs save public variant names only: `baseline` and `improved_with_se`.
- Checkpoint loading remains backward-compatible with older checkpoints that stored legacy variant metadata, but state-dict loading is strict.

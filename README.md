# UTKFace Age & Gender Prediction Project

A complete local Python project for **age regression** and **gender classification** on **UTKFace**.

## Main features

- Age as **regression**
- Gender as **binary classification**
- **Multitask CNN** trained from scratch (no pretrained weights in the main model)
- Proper **train / val / test split** saved to CSV
- Full training, evaluation, and inference pipeline
- **OpenCV benchmark** comparison on the **same test set**
- **Naive baselines** on the same test set
- Training curves, scatter plots, residual plots, confusion matrix, ROC curve, benchmark charts
- Results exported to **CSV / JSON / PNG**
- Simple **ablation experiment** runner

## Dataset expectation

The project assumes UTKFace filenames follow this pattern:

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
  models/
    opencv/
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
    download_opencv_models.py
    benchmarks_opencv.py
    run_ablation.py
  models/
    opencv/
  outputs/
    logs/
    models/
    plots/
    results/
```

## Setup

### 1) Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training, make sure your `torch` installation is the CUDA-enabled one for your machine.

## Python and device notes

- Recommended Python: **3.10 or 3.11**
- The training and evaluation scripts automatically use **GPU** when `torch.cuda.is_available()` is `True`; otherwise they fall back to CPU.
- To use an NVIDIA GPU, install a **CUDA-enabled PyTorch build** that matches your system. The exact install command can differ by CUDA version, so check the official PyTorch selector if needed.
- This project intentionally avoids `torchvision` so you do not run into the common `torchvision::nms does not exist` compatibility issue.

## Step-by-step workflow

### Step 1: Build the metadata CSV and split file

```bash
python src/prepare_data.py --data_dir data/UTKFace --output_csv outputs/results/utkface_metadata.csv --split_csv outputs/results/utkface_splits.csv
```

This creates:

- `outputs/results/utkface_metadata.csv`
- `outputs/results/utkface_splits.csv`
- dataset distribution plots in `outputs/plots/prepare_data/`

### Step 2: Train the main model

Baseline from-scratch model:

```bash
python src/train.py --split_csv outputs/results/utkface_splits.csv --experiment_name baseline_cnn --variant baseline --epochs 25 --batch_size 64 --image_size 128
```

Improved hybrid model:

```bash
python src/train.py --split_csv outputs/results/utkface_splits.csv --experiment_name improved_cnn --variant improved --epochs 35 --batch_size 64 --image_size 128 --dropout 0.35 --age_loss huber --use_se
```

### Step 3: Evaluate the trained model on the test split

```bash
python src/evaluate.py --split_csv outputs/results/utkface_splits.csv --checkpoint outputs/models/improved_cnn/best_model.pt --experiment_name improved_cnn
```

### Step 4: Run naive baselines on the same test split

```bash
python src/benchmarks_naive.py --split_csv outputs/results/utkface_splits.csv --experiment_name naive_baselines
```

### Step 5: Download OpenCV benchmark models

```bash
python src/download_opencv_models.py --model_dir models/opencv
```

### Step 6: Run the OpenCV benchmark on the same test split

```bash
python src/benchmarks_opencv.py --split_csv outputs/results/utkface_splits.csv --model_dir models/opencv --experiment_name opencv_benchmark
```

### Step 7: Run the ablation suite

```bash
python src/run_ablation.py --split_csv outputs/results/utkface_splits.csv
```

## Important outputs

### Training outputs

- `outputs/models/<experiment_name>/best_model.pt`
- `outputs/logs/<experiment_name>/history.csv`
- `outputs/plots/<experiment_name>/training_total_loss.png`
- `outputs/plots/<experiment_name>/training_age_mae.png`
- `outputs/plots/<experiment_name>/training_gender_accuracy.png`

### Evaluation outputs

- `outputs/results/<experiment_name>/test_predictions.csv`
- `outputs/results/<experiment_name>/metrics.json`
- `outputs/results/<experiment_name>/metrics_summary.csv`
- `outputs/results/<experiment_name>/group_metrics.csv`
- `outputs/plots/<experiment_name>/age_true_vs_pred.png`
- `outputs/plots/<experiment_name>/age_residual_hist.png`
- `outputs/plots/<experiment_name>/age_residual_scatter.png`
- `outputs/plots/<experiment_name>/gender_confusion_matrix.png`
- `outputs/plots/<experiment_name>/gender_roc_curve.png`
- `outputs/plots/<experiment_name>/sample_predictions.png`

### Benchmark outputs

- `outputs/results/naive_baselines/metrics_summary.csv`
- `outputs/results/opencv_benchmark/metrics_summary.csv`
- `outputs/results/benchmark_comparison.csv`
- `outputs/plots/benchmark_comparison/benchmark_bar_age_mae.png`
- `outputs/plots/benchmark_comparison/benchmark_bar_gender_accuracy.png`

## Fair OpenCV comparison note

The OpenCV age model predicts **age buckets**, not exact ages. This project compares fairly in two ways:

1. **Bucket accuracy**: both methods are converted to the same 8 age buckets.
2. **Approximate regression MAE**: OpenCV bucket outputs are converted to bucket midpoints before computing MAE.

Use bucket accuracy as the fairest direct comparison.

## Good report points

When you interpret the results, focus on:

- **Age MAE** for exact-age performance
- **Gender accuracy / F1** for classification quality
- Whether the improved CNN beats the baseline CNN
- Whether the CNN beats the naive baselines
- Whether the CNN beats OpenCV on UTKFace test data
- Whether OpenCV performs worse partly because it was trained for age **groups**, not exact age regression



## Notes for the fixed version
- Added an auxiliary age-bucket head for hybrid regression + bucket supervision.
- Added optional SE blocks with `--use_se`.
- All matplotlib plots now include grids.
- CSV export now includes `age_bucket_excel` to stop spreadsheet apps from turning `0-2` into dates.
- Data split now uses age-strata + gender stratification and writes a split distribution summary.
- GPU is used automatically when a CUDA-enabled PyTorch install is available.
- `torchvision` was removed from the project dependency chain to avoid version-mismatch import failures.

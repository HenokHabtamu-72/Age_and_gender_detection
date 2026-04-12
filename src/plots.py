from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

try:
    from .utils import SPLIT_DISPLAY_ORDER, ensure_dir
except ImportError:
    from utils import SPLIT_DISPLAY_ORDER, ensure_dir


def _apply_grid(ax) -> None:
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)


def save_plot(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_age_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['age'], bins=30)
    ax.set_title('UTKFace age distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    _apply_grid(ax)
    save_plot(fig, path)


def plot_gender_distribution(df: pd.DataFrame, path: Path) -> None:
    label_map = {0: 'male', 1: 'female'}
    counts = df['gender'].map(label_map).value_counts()
    ordered_labels = [label for label in ['male', 'female'] if label in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ordered_labels, [int(counts[label]) for label in ordered_labels])
    ax.set_title('UTKFace gender distribution')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    _apply_grid(ax)
    save_plot(fig, path)


def plot_split_distribution(df: pd.DataFrame, path: Path) -> None:
    counts = df['split'].value_counts()
    ordered_splits = [split for split in SPLIT_DISPLAY_ORDER if split in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ordered_splits, [int(counts[split]) for split in ordered_splits])
    ax.set_title('Train / val / test split counts')
    ax.set_xlabel('Split')
    ax.set_ylabel('Count')
    _apply_grid(ax)
    save_plot(fig, path)


def plot_split_age_boxplot(df: pd.DataFrame, path: Path) -> None:
    order = [s for s in SPLIT_DISPLAY_ORDER if s in df['split'].unique()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([df.loc[df['split'] == split_name, 'age'].to_numpy() for split_name in order], labels=order)
    ax.set_title('Age distribution by split')
    ax.set_xlabel('Split')
    ax.set_ylabel('Age')
    _apply_grid(ax)
    save_plot(fig, path)



def plot_history(history_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    plot_specs = [
        ('Training total loss', 'training_total_loss.png', [('train_total_loss', 'train_total_loss'), ('val_total_loss', 'val_total_loss')], 'loss'),
        ('Age MAE', 'training_age_mae.png', [('train_age_mae', 'train_age_mae'), ('val_age_mae', 'val_age_mae')], 'mae'),
        ('Gender accuracy', 'training_gender_accuracy.png', [('train_gender_accuracy', 'train_gender_accuracy'), ('val_gender_accuracy', 'val_gender_accuracy')], 'accuracy'),
    ]
    for title, filename, series, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col, label in series:
            ax.plot(history_df['epoch'], history_df[col], label=label)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        _apply_grid(ax)
        save_plot(fig, output_dir / filename)



def plot_age_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4)
    line_min = min(y_true.min(), y_pred.min())
    line_max = max(y_true.max(), y_pred.max())
    ax.plot([line_min, line_max], [line_min, line_max], linestyle='--')
    ax.set_title('True age vs predicted age')
    ax.set_xlabel('True age')
    ax.set_ylabel('Predicted age')
    _apply_grid(ax)
    save_plot(fig, path)



def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30)
    ax.set_title('Age residual histogram')
    ax.set_xlabel('Predicted age - true age')
    ax.set_ylabel('Count')
    _apply_grid(ax)
    save_plot(fig, path)



def plot_residual_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true, residuals, alpha=0.4)
    ax.axhline(0, linestyle='--')
    ax.set_title('Age residuals vs true age')
    ax.set_xlabel('True age')
    ax.set_ylabel('Residual')
    _apply_grid(ax)
    save_plot(fig, path)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path, labels=None, title: str = 'Confusion matrix') -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    _apply_grid(ax)
    save_plot(fig, path)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title('Gender ROC curve')
    _apply_grid(ax)
    save_plot(fig, path)


def plot_sample_predictions(df_pred: pd.DataFrame, path: Path, max_samples: int = 12) -> None:
    import matplotlib.image as mpimg
    sample_df = df_pred.sample(min(max_samples, len(df_pred)), random_state=42).reset_index(drop=True)
    cols = 3
    rows = int(np.ceil(len(sample_df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes[len(sample_df):]:
        ax.axis('off')
    for idx, row in sample_df.iterrows():
        ax = axes[idx]
        try:
            ax.imshow(mpimg.imread(row['image_path']))
        except Exception:
            ax.text(0.5, 0.5, 'Image load failed', ha='center', va='center')
        ax.axis('off')
        ax.set_title(
            f"T_age={row['true_age']}, P_age={row['pred_age']:.1f}\n"
            f"T_gender={int(row['true_gender'])}, P_gender={int(row['pred_gender'])}\n"
            f"G_prob={row['pred_gender_prob']:.2f}, abs_err={abs(row['pred_age'] - row['true_age']):.1f}",
            fontsize=9,
        )
    save_plot(fig, path)


def plot_benchmark_bars(benchmark_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    for metric, fname, title in [
        ('age_mae', 'benchmark_bar_age_mae.png', 'Benchmark comparison: age MAE (lower is better)'),
        ('gender_accuracy', 'benchmark_bar_gender_accuracy.png', 'Benchmark comparison: gender accuracy (higher is better)'),
    ]:
        if metric not in benchmark_df.columns:
            continue
        plot_df = benchmark_df[['method', metric]].dropna().copy()
        if plot_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(plot_df['method'], plot_df[metric])
        ax.set_title(title)
        ax.set_xlabel('Method')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=30)
        _apply_grid(ax)
        save_plot(fig, output_dir / fname)

import argparse
import math

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    LOGS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
)
from dataset import UTKFaceDataset
from model import MultiTaskCNN
from plots import plot_history
from utils import compute_age_bucket_accuracy, count_parameters, ensure_dir, save_json, set_seed


def get_losses(age_loss_name: str = "huber"):
    if age_loss_name == "huber":
        age_loss_fn = nn.SmoothL1Loss(beta=3.0)
    elif age_loss_name == "mse":
        age_loss_fn = nn.MSELoss()
    elif age_loss_name == "mae":
        age_loss_fn = nn.L1Loss()
    else:
        raise ValueError("Unsupported age loss")

    gender_loss_fn = nn.BCEWithLogitsLoss()
    age_bucket_loss_fn = nn.CrossEntropyLoss()
    return age_loss_fn, gender_loss_fn, age_bucket_loss_fn


def run_one_epoch(model, loader, optimizer, device, age_loss_fn, gender_loss_fn, age_bucket_loss_fn, age_weight, gender_weight, age_bucket_weight, train=True):
    model.train(train)
    total_loss_sum = 0.0
    total_age_loss_sum = 0.0
    total_gender_loss_sum = 0.0
    total_age_bucket_loss_sum = 0.0
    total_samples = 0

    age_true_all = []
    age_pred_all = []
    gender_true_all = []
    gender_pred_all = []
    age_bucket_true_all = []
    age_bucket_pred_all = []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        age_true = batch["age"].to(device)
        gender_true = batch["gender"].to(device)
        age_bucket_true = batch["age_bucket"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        age_pred = outputs["age"]
        gender_logit = outputs["gender_logit"]
        age_bucket_logits = outputs["age_bucket_logits"]

        age_loss = age_loss_fn(age_pred, age_true)
        gender_loss = gender_loss_fn(gender_logit, gender_true)
        age_bucket_loss = age_bucket_loss_fn(age_bucket_logits, age_bucket_true)
        total_loss = age_weight * age_loss + gender_weight * gender_loss + age_bucket_weight * age_bucket_loss

        if train:
            total_loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss_sum += float(total_loss.item()) * batch_size
        total_age_loss_sum += float(age_loss.item()) * batch_size
        total_gender_loss_sum += float(gender_loss.item()) * batch_size
        total_age_bucket_loss_sum += float(age_bucket_loss.item()) * batch_size

        age_true_np = age_true.detach().cpu().numpy()
        age_pred_np = age_pred.detach().cpu().numpy()
        gender_true_np = gender_true.detach().cpu().numpy().astype(int)
        gender_pred_np = (torch.sigmoid(gender_logit).detach().cpu().numpy() >= 0.5).astype(int)
        age_bucket_true_np = age_bucket_true.detach().cpu().numpy().astype(int)
        age_bucket_pred_np = age_bucket_logits.argmax(dim=1).detach().cpu().numpy().astype(int)

        age_true_all.extend(age_true_np.tolist())
        age_pred_all.extend(age_pred_np.tolist())
        gender_true_all.extend(gender_true_np.tolist())
        gender_pred_all.extend(gender_pred_np.tolist())
        age_bucket_true_all.extend(age_bucket_true_np.tolist())
        age_bucket_pred_all.extend(age_bucket_pred_np.tolist())

    age_true_all = pd.Series(age_true_all, dtype=float)
    age_pred_all = pd.Series(age_pred_all, dtype=float)
    gender_true_all = pd.Series(gender_true_all, dtype=int)
    gender_pred_all = pd.Series(gender_pred_all, dtype=int)
    age_bucket_true_all = pd.Series(age_bucket_true_all, dtype=int)
    age_bucket_pred_all = pd.Series(age_bucket_pred_all, dtype=int)

    metrics = {
        "total_loss": total_loss_sum / max(total_samples, 1),
        "age_loss": total_age_loss_sum / max(total_samples, 1),
        "gender_loss": total_gender_loss_sum / max(total_samples, 1),
        "age_bucket_loss": total_age_bucket_loss_sum / max(total_samples, 1),
        "age_mae": float((age_true_all - age_pred_all).abs().mean()),
        "gender_accuracy": float((gender_true_all == gender_pred_all).mean()),
        "age_bucket_accuracy": float((age_bucket_true_all == age_bucket_pred_all).mean()),
        "regression_bucket_accuracy": compute_age_bucket_accuracy(age_true_all.to_numpy(), age_pred_all.to_numpy()),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="improved_cnn")
    parser.add_argument("--variant", type=str, default="improved", choices=["baseline", "improved"])
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--use_se", action="store_true")
    parser.add_argument("--age_weight", type=float, default=1.0)
    parser.add_argument("--gender_weight", type=float, default=1.0)
    parser.add_argument("--age_bucket_weight", type=float, default=0.30)
    parser.add_argument("--age_loss", type=str, default="huber", choices=["huber", "mse", "mae"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split_df = pd.read_csv(args.split_csv)
    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()

    if "age_bucket_idx" not in train_df.columns:
        from utils import age_to_bucket_index
        train_df["age_bucket_idx"] = train_df["age"].apply(age_to_bucket_index)
        val_df["age_bucket_idx"] = val_df["age"].apply(age_to_bucket_index)

    train_ds = UTKFaceDataset(train_df, image_size=args.image_size, train=True)
    val_ds = UTKFaceDataset(val_df, image_size=args.image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = MultiTaskCNN(variant=args.variant, dropout=args.dropout, use_se=args.use_se).to(device)
    age_loss_fn, gender_loss_fn, age_bucket_loss_fn = get_losses(args.age_loss)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    model_dir = MODELS_DIR / args.experiment_name
    log_dir = LOGS_DIR / args.experiment_name
    plot_dir = PLOTS_DIR / args.experiment_name
    ensure_dir(model_dir)
    ensure_dir(log_dir)
    ensure_dir(plot_dir)

    config_dict = vars(args).copy()
    config_dict["device"] = str(device)
    config_dict["parameter_count"] = count_parameters(model)
    save_json(config_dict, model_dir / "train_config.json")

    best_val_age_mae = math.inf
    history_rows = []
    patience = 7
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = run_one_epoch(
            model, train_loader, optimizer, device, age_loss_fn, gender_loss_fn, age_bucket_loss_fn,
            args.age_weight, args.gender_weight, args.age_bucket_weight, train=True
        )
        with torch.no_grad():
            val_metrics = run_one_epoch(
                model, val_loader, optimizer, device, age_loss_fn, gender_loss_fn, age_bucket_loss_fn,
                args.age_weight, args.gender_weight, args.age_bucket_weight, train=False
            )

        scheduler.step(val_metrics["total_loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_total_loss": train_metrics["total_loss"],
            "train_age_mae": train_metrics["age_mae"],
            "train_gender_accuracy": train_metrics["gender_accuracy"],
            "train_age_bucket_accuracy": train_metrics["age_bucket_accuracy"],
            "train_regression_bucket_accuracy": train_metrics["regression_bucket_accuracy"],
            "val_total_loss": val_metrics["total_loss"],
            "val_age_mae": val_metrics["age_mae"],
            "val_gender_accuracy": val_metrics["gender_accuracy"],
            "val_age_bucket_accuracy": val_metrics["age_bucket_accuracy"],
            "val_regression_bucket_accuracy": val_metrics["regression_bucket_accuracy"],
        }
        history_rows.append(row)
        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(log_dir / "history.csv", index=False)
        plot_history(history_df, plot_dir)

        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

        if val_metrics["age_mae"] < best_val_age_mae:
            best_val_age_mae = val_metrics["age_mae"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "variant": args.variant,
                    "dropout": args.dropout,
                    "image_size": args.image_size,
                    "use_se": args.use_se,
                    "epoch": epoch,
                    "best_val_age_mae": best_val_age_mae,
                },
                model_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best validation age MAE: {best_val_age_mae:.4f}")
    print(f"Best model saved to: {model_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()

import argparse
import math

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from .config import AGE_MAX, AGE_MIN, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_IMAGE_SIZE, DEFAULT_LR, DEFAULT_NUM_WORKERS, DEFAULT_SEED, DEFAULT_WEIGHT_DECAY, LOGS_DIR, MODELS_DIR, PLOTS_DIR, resolve_project_path
    from .dataset import UTKFaceDataset
    from .model import MultiTaskCNN, normalize_public_variant
    from .plots import plot_history
    from .utils import count_parameters, ensure_dir, save_json, set_seed
except ImportError:
    from config import AGE_MAX, AGE_MIN, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_IMAGE_SIZE, DEFAULT_LR, DEFAULT_NUM_WORKERS, DEFAULT_SEED, DEFAULT_WEIGHT_DECAY, LOGS_DIR, MODELS_DIR, PLOTS_DIR, resolve_project_path
    from dataset import UTKFaceDataset
    from model import MultiTaskCNN, normalize_public_variant
    from plots import plot_history
    from utils import count_parameters, ensure_dir, save_json, set_seed



def get_losses(age_loss_name: str = 'huber'):
    if age_loss_name == 'huber':
        age_loss_fn = nn.SmoothL1Loss(beta=3.0)
    elif age_loss_name == 'mse':
        age_loss_fn = nn.MSELoss()
    elif age_loss_name == 'mae':
        age_loss_fn = nn.L1Loss()
    else:
        raise ValueError("age_loss must be one of: 'huber', 'mse', 'mae'")
    return age_loss_fn, nn.BCEWithLogitsLoss()



def resolve_use_se(variant: str) -> bool:
    return normalize_public_variant(variant) == 'improved_with_se'



def run_one_epoch(model, loader, optimizer, device, age_loss_fn, gender_loss_fn, age_weight, gender_weight, train=True):
    model.train(train)
    total_loss_sum = 0.0
    total_age_loss_sum = 0.0
    total_gender_loss_sum = 0.0
    total_samples = 0
    age_true_all, age_pred_all, gender_true_all, gender_pred_all = [], [], [], []
    for batch in loader:
        images = batch['image'].to(device)
        age_true = batch['age'].to(device)
        gender_true = batch['gender'].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        age_pred = outputs['age']
        gender_logit = outputs['gender_logit']
        age_loss = age_loss_fn(age_pred, age_true)
        gender_loss = gender_loss_fn(gender_logit, gender_true)
        total_loss = age_weight * age_loss + gender_weight * gender_loss
        if train:
            total_loss.backward()
            optimizer.step()
        batch_size = images.size(0)
        total_samples += batch_size
        total_loss_sum += float(total_loss.item()) * batch_size
        total_age_loss_sum += float(age_loss.item()) * batch_size
        total_gender_loss_sum += float(gender_loss.item()) * batch_size
        age_true_all.extend(age_true.detach().cpu().numpy().tolist())
        age_pred_all.extend(age_pred.detach().cpu().numpy().tolist())
        gender_true_all.extend(gender_true.detach().cpu().numpy().astype(int).tolist())
        gender_pred_all.extend((torch.sigmoid(gender_logit).detach().cpu().numpy() >= 0.5).astype(int).tolist())
    age_true_all = pd.Series(age_true_all, dtype=float)
    age_pred_all = pd.Series(age_pred_all, dtype=float).clip(lower=AGE_MIN, upper=AGE_MAX)
    gender_true_all = pd.Series(gender_true_all, dtype=int)
    gender_pred_all = pd.Series(gender_pred_all, dtype=int)
    return {
        'total_loss': total_loss_sum / max(total_samples, 1),
        'age_loss': total_age_loss_sum / max(total_samples, 1),
        'gender_loss': total_gender_loss_sum / max(total_samples, 1),
        'age_mae': float((age_true_all - age_pred_all).abs().mean()),
        'gender_accuracy': float((gender_true_all == gender_pred_all).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_csv', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, default='baseline_cnn')
    parser.add_argument('--variant', type=str, default='baseline', choices=['baseline', 'improved_with_se'])
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument('--dropout', type=float, default=0.30)
    parser.add_argument('--age_weight', type=float, default=1.0)
    parser.add_argument('--gender_weight', type=float, default=1.0)
    parser.add_argument('--age_loss', type=str, default='huber', choices=['huber', 'mse', 'mae'])
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    args.variant = normalize_public_variant(args.variant)
    use_se = resolve_use_se(args.variant)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    split_df = pd.read_csv(resolve_project_path(args.split_csv))
    train_df = split_df[split_df['split'] == 'train'].copy()
    val_df = split_df[split_df['split'] == 'val'].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("split_csv must contain both 'train' and 'val' rows.")
    train_ds = UTKFaceDataset(train_df, image_size=args.image_size, train=True)
    val_ds = UTKFaceDataset(val_df, image_size=args.image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = MultiTaskCNN(variant=args.variant, dropout=args.dropout, use_se=use_se).to(device)
    age_loss_fn, gender_loss_fn = get_losses(args.age_loss)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    model_dir = MODELS_DIR / args.experiment_name
    log_dir = LOGS_DIR / args.experiment_name
    plot_dir = PLOTS_DIR / args.experiment_name
    ensure_dir(model_dir)
    ensure_dir(log_dir)
    ensure_dir(plot_dir)

    config_dict = vars(args).copy()
    config_dict['split_csv'] = str(resolve_project_path(args.split_csv))
    config_dict['device'] = str(device)
    config_dict['parameter_count'] = count_parameters(model)
    config_dict['use_se'] = use_se
    save_json(config_dict, model_dir / 'train_config.json')

    best_val_total_loss = math.inf
    history_rows = []
    patience = 7
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, age_loss_fn, gender_loss_fn, args.age_weight, args.gender_weight, train=True)
        with torch.no_grad():
            val_metrics = run_one_epoch(model, val_loader, optimizer, device, age_loss_fn, gender_loss_fn, args.age_weight, args.gender_weight, train=False)
        scheduler.step(val_metrics['total_loss'])
        row = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_total_loss': train_metrics['total_loss'],
            'train_age_mae': train_metrics['age_mae'],
            'train_gender_accuracy': train_metrics['gender_accuracy'],
            'val_total_loss': val_metrics['total_loss'],
            'val_age_mae': val_metrics['age_mae'],
            'val_gender_accuracy': val_metrics['gender_accuracy'],
        }
        history_rows.append(row)
        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(log_dir / 'history.csv', index=False)
        plot_history(history_df, plot_dir)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        if val_metrics['total_loss'] < best_val_total_loss:
            best_val_total_loss = val_metrics['total_loss']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'variant': args.variant,
                'dropout': args.dropout,
                'image_size': args.image_size,
                'use_se': use_se,
                'epoch': epoch,
                'best_val_total_loss': best_val_total_loss,
                'best_val_age_mae': val_metrics['age_mae'],
                'best_val_gender_accuracy': val_metrics['gender_accuracy'],
            }, model_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered.')
                break
    print(f'Best validation total loss: {best_val_total_loss:.4f}')
    print(f"Best model saved to: {model_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()

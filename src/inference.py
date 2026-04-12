import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image

try:
    from .config import AGE_MAX, AGE_MIN, resolve_project_path
    from .model import MultiTaskCNN
    from .simple_transforms import SimpleImageTransform
except ImportError:
    from config import AGE_MAX, AGE_MIN, resolve_project_path
    from model import MultiTaskCNN
    from simple_transforms import SimpleImageTransform



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



def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    variant, use_se = resolve_checkpoint_model_options(ckpt)
    model = MultiTaskCNN(variant=variant, dropout=ckpt['dropout'], use_se=use_se).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model, ckpt



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model, ckpt = load_model(resolve_project_path(args.checkpoint), device)
    transform = SimpleImageTransform(image_size=ckpt['image_size'], train=False)
    image = Image.open(resolve_project_path(args.image)).convert('RGB')
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred_age = float(max(AGE_MIN, min(AGE_MAX, out['age'].item())))
        pred_gender_prob = float(out['gender_prob'].item())
        pred_gender = 1 if pred_gender_prob >= 0.5 else 0
    print(f'Predicted age:             {pred_age:.2f}')
    print(f'Predicted gender:          {pred_gender} (0=male, 1=female)')
    print(f'Gender probability:        {pred_gender_prob:.4f}')


if __name__ == '__main__':
    main()

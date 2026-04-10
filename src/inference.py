import argparse
from pathlib import Path

import torch
from PIL import Image
from simple_transforms import SimpleImageTransform

from config import AGE_BUCKET_LABELS, AGE_MAX, AGE_MIN
from model import MultiTaskCNN
from utils import age_to_bucket_index


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MultiTaskCNN(
        variant=ckpt["variant"],
        dropout=ckpt["dropout"],
        use_se=ckpt.get("use_se", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, ckpt = load_model(Path(args.checkpoint), device)

    transform = SimpleImageTransform(image_size=ckpt["image_size"], train=False)

    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        pred_age = float(max(AGE_MIN, min(AGE_MAX, out["age"].item())))
        pred_gender_prob = float(out["gender_prob"].item())
        pred_gender = 1 if pred_gender_prob >= 0.5 else 0
        pred_bucket = age_to_bucket_index(pred_age)

    print(f"Predicted age (regression): {pred_age:.2f}")
    print(f"Predicted age bucket:       {AGE_BUCKET_LABELS[pred_bucket]}")
    print(f"Predicted gender:           {pred_gender} (0=male, 1=female)")
    print(f"Gender probability:         {pred_gender_prob:.4f}")


if __name__ == "__main__":
    main()

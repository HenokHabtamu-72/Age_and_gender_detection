import torch
import torch.nn as nn

from config import NUM_AGE_BUCKETS


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, use_se=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_se:
            layers.append(SEModule(out_channels))
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MultiTaskCNN(nn.Module):
    def __init__(self, variant: str = "baseline", dropout: float = 0.3, use_se: bool = False):
        super().__init__()
        if variant not in {"baseline", "improved"}:
            raise ValueError("variant must be 'baseline' or 'improved'")

        if variant == "baseline":
            channels = [32, 64, 128, 256]
        else:
            channels = [32, 64, 128, 256, 384]

        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch, pool=True, use_se=use_se))
            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.age_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.gender_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.age_bucket_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_AGE_BUCKETS),
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.global_pool(feats)
        feats = self.shared_fc(feats)
        age = self.age_head(feats).squeeze(1)
        gender_logit = self.gender_head(feats).squeeze(1)
        age_bucket_logits = self.age_bucket_head(feats)
        age_bucket_prob = torch.softmax(age_bucket_logits, dim=1)
        return {
            "age": age,
            "gender_logit": gender_logit,
            "gender_prob": torch.sigmoid(gender_logit),
            "age_bucket_logits": age_bucket_logits,
            "age_bucket_prob": age_bucket_prob,
        }

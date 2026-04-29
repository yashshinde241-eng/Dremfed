"""
DermFed – utils.py
Shared dataset class, model factory, and training helpers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES = 7
IMG_SIZE    = 224
BATCH_SIZE  = 32

LABEL_MAP = {
    "akiec": 0,
    "bcc":   1,
    "bkl":   2,
    "df":    3,
    "mel":   4,
    "nv":    5,
    "vasc":  6,
}

CLASS_NAMES = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevi (nv)",
    "Vascular Lesions (vasc)",
]

CLASS_COLORS = [
    "#FF6B6B",  # akiec – red
    "#FF8E53",  # bcc   – orange
    "#FFD93D",  # bkl   – yellow
    "#6BCB77",  # df    – green
    "#4D96FF",  # mel   – blue
    "#A855F7",  # nv    – purple
    "#F472B6",  # vasc  – pink
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ───────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Dataset ──────────────────────────────────────────────────────────────────
class SkinLesionDataset(Dataset):
    """
    Reads a silo's metadata.csv.
    Expected columns: file_path, label  (int 0-6)
    """

    def __init__(self, csv_path: str | Path, transform=None) -> None:
        self.df        = pd.read_csv(csv_path)
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row   = self.df.iloc[idx]
        img   = Image.open(row["file_path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


def get_loaders(silo_dir: str | Path,
                batch_size: int = BATCH_SIZE,
                val_split: float = 0.15) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for a given hospital silo directory."""
    from sklearn.model_selection import train_test_split

    csv   = Path(silo_dir) / "metadata.csv"
    df    = pd.read_csv(csv)
    train_df, val_df = train_test_split(df, test_size=val_split,
                                        stratify=df["label"], random_state=42)

    # Write temp CSVs so Dataset can load them
    tmp_dir = Path(silo_dir) / ".splits"
    tmp_dir.mkdir(exist_ok=True)
    train_csv = tmp_dir / "train.csv"
    val_csv   = tmp_dir / "val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_ds = SkinLesionDataset(train_csv, transform=TRAIN_TRANSFORM)
    val_ds   = SkinLesionDataset(val_csv,   transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES,
                freeze_backbone: bool = False) -> nn.Module:
    """MobileNetV2 with a custom classification head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimiser: torch.optim.Optimizer,
                    criterion: nn.Module) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs     = model(images)
        loss        = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, image: Image.Image) -> Tuple[int, float, list[float]]:
    """
    Classify a PIL image.
    Returns (predicted_class_idx, confidence, all_probabilities).
    """
    model.eval()
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    pred   = int(torch.argmax(logits, dim=1).item())
    return pred, probs[pred], probs

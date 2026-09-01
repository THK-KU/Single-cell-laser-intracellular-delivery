"""Train a U-Net to segment cells in circular-masked well images.

Inputs are the masked well images and their hand-drawn binary masks, paired
by filename::

    <data>/BF_circle/<name>.jpg      image
    <data>/mask_circle/<name>.jpg    mask (white = cell)

Typical run::

    python unet_train.py --images data/train/BF_circle \
                         --masks  data/train/mask_circle \
                         --output models/unet.pth

Augmentation (flips, quarter-turns, brightness jitter) is applied to the
whole dataset, so the validation loss is measured on augmented images too
and is comparable in difficulty to the training loss.

The best checkpoint by validation loss is written to ``--output``, alongside
a JSON file recording the split, the seed and the loss history. Segmentation
accuracy (Dice) is measured separately, on unaugmented images.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with group normalisation."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Three-level U-Net returning logits (no sigmoid; see BCEDiceLoss)."""

    def __init__(self, in_channels: int = 1, base_ch: int = 64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Soft Dice loss, computed over the whole batch."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(
            logits, targets
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CellDataset(Dataset):
    """Well images paired with binary cell masks of the same filename."""

    def __init__(
        self,
        img_dir: Path,
        mask_dir: Path,
        names: list[str],
        crop_size: tuple[int, int] = (448, 448),
        augment: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.names = names
        self.crop_size = crop_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.names)

    def _augment(self, img, mask):
        """Flips and quarter-turns applied to both; brightness to the image."""
        if random.random() < 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() < 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)
        k = random.randint(0, 3)
        if k:
            img, mask = TF.rotate(img, 90 * k), TF.rotate(mask, 90 * k)
        if random.random() < 0.3:
            img = TF.adjust_brightness(img, 0.8 + 0.4 * random.random())
        return img, mask

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = Image.open(self.img_dir / name).convert("L")
        mask = Image.open(self.mask_dir / name).convert("L")

        img = TF.center_crop(img, self.crop_size)
        mask = TF.center_crop(mask, self.crop_size)

        if self.augment:
            img, mask = self._augment(img, mask)

        return self.to_tensor(img), (self.to_tensor(mask) > 0.5).float()


def paired_names(img_dir: Path, mask_dir: Path) -> list[str]:
    """Filenames present in both directories, sorted.

    Images without a mask (or vice versa) are excluded and counted, so a
    missing annotation does not silently shrink the training set.
    """
    images = {p.name for p in img_dir.iterdir() if p.is_file()}
    masks = {p.name for p in mask_dir.iterdir() if p.is_file()}
    paired = sorted(images & masks)

    for label, missing in (("mask", images - masks), ("image", masks - images)):
        if missing:
            print(f"warning: {len(missing)} file(s) with no {label}", file=sys.stderr)
    return paired


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed every generator the training loop draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    """Everything that determines the outcome of a run."""

    seed: int = 42
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-3
    val_ratio: float = 0.1
    crop_size: int = 448
    base_channels: int = 64
    bce_weight: float = 0.5
    dice_weight: float = 0.5


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    """One pass over ``loader``. Trains when ``optimizer`` is given."""
    training = optimizer is not None
    model.train(training)
    total = 0.0

    with torch.set_grad_enabled(training):
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * imgs.size(0)

    return total / len(loader.dataset)


def train(
    img_dir: Path,
    mask_dir: Path,
    output: Path,
    config: TrainConfig,
) -> None:
    set_seed(config.seed)

    names = paired_names(img_dir, mask_dir)
    if not names:
        raise SystemExit(f"no paired files between {img_dir} and {mask_dir}")

    n_val = int(len(names) * config.val_ratio)
    perm = torch.randperm(
        len(names), generator=torch.Generator().manual_seed(config.seed)
    ).tolist()
    val_idx, train_idx = sorted(perm[:n_val]), sorted(perm[n_val:])

    # Augmentation is applied to the whole dataset: both splits index into a
    # single dataset built with augment=True, so the validation loss reflects
    # the same transforms as the training loss.
    dataset = CellDataset(
        img_dir, mask_dir, names, (config.crop_size, config.crop_size), augment=True
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False
    )
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UNet(in_channels=1, base_ch=config.base_channels).to(device)
    criterion = BCEDiceLoss(config.bce_weight, config.dice_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(config.epochs):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss})

        marker = ""
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch + 1
            torch.save(model.state_dict(), output)
            marker = "  <- saved"

        print(
            f"[{epoch + 1:>3}/{config.epochs}] "
            f"train {train_loss:.4f} | val {val_loss:.4f}{marker}"
        )

    metadata = {
        "config": asdict(config),
        "images": str(img_dir),
        "masks": str(mask_dir),
        "checkpoint": output.name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_files": [names[i] for i in train_idx],
        "val_files": [names[i] for i in val_idx],
        "history": history,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nBest epoch {best_epoch}, val loss {best_val:.4f}")
    print(f"Checkpoint: {output}\nRun record: {metadata_path}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train a U-Net to segment cells in well images."
    )
    parser.add_argument("--images", type=Path, required=True, help="well images")
    parser.add_argument("--masks", type=Path, required=True, help="binary cell masks")
    parser.add_argument(
        "--output", type=Path, required=True, help="checkpoint path (.pth)"
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--crop-size", type=int, default=defaults.crop_size)
    parser.add_argument("--base-channels", type=int, default=defaults.base_channels)
    args = parser.parse_args(argv)

    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        crop_size=args.crop_size,
        base_channels=args.base_channels,
    )
    train(args.images, args.masks, args.output, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
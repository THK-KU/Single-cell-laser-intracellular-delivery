"""Predict cell masks with a trained U-Net.

Reads the circle-masked well images and writes one binary mask per well::

    9_after/BF_circle/BF_split_A_0_0_0_0_5_1.0.jpg
      -> 9_after/result_mask/BF_split_A_0_0_0_0_5_1.0_pred.png

Run over the whole dataset::

    python unet_predict.py --root data/raw --model models/unet.pth

or over a single directory::

    python unet_predict.py --input  data/raw/9_after/BF_circle \
                           --output data/raw/9_after/result_mask \
                           --model  models/unet.pth

Predictions are thresholded at ``--threshold`` and connected components
smaller than ``--min-area`` pixels are discarded. Both values determine what
counts as a cell, so they are recorded in the printed summary.

The model definition is imported from ``unet_train`` so that training and
inference can never drift apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from unet_train import UNet

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


DEFAULT_MIN_AREA = 5000
DEFAULT_THRESHOLD = 0.5
CROP_SIZE = 448


class InferenceDataset(Dataset):
    """Well images, centre-cropped to the size the model was trained on."""

    def __init__(self, img_dir: Path, crop_size: int = CROP_SIZE):
        self.img_dir = Path(img_dir)
        self.names = sorted(
            p.name for p in self.img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        self.crop_size = (crop_size, crop_size)
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = Image.open(self.img_dir / name).convert("L")
        img = TF.center_crop(img, self.crop_size)
        return self.to_tensor(img), name


def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components below ``min_area`` pixels.

    Contours are filled, so holes inside a retained object are filled in.
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean = np.zeros_like(mask_u8)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, thickness=-1)
    return (clean > 0).astype(np.float32)


def make_overlay(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Tint the predicted region red over the input image, for inspection."""
    gray_u8 = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    overlay = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    if mask.sum() == 0:
        return overlay

    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    selected = mask.astype(bool)
    overlay[selected] = cv2.addWeighted(
        overlay[selected], 1 - alpha, red[selected], alpha, 0
    )
    return overlay


def load_model(model_path: Path, device: torch.device) -> UNet:
    model = UNet(in_channels=1, base_ch=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_directory(
    model: UNet,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    overlay_dir: Path | None = None,
) -> tuple[int, int]:
    """Predict every well in ``input_dir``.

    Returns (number written, number whose mask came out empty).
    """
    dataset = InferenceDataset(input_dir)
    if not len(dataset):
        print(f"  no images in {input_dir}", file=sys.stderr)
        return 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    written = 0
    empty = 0

    with torch.no_grad():
        for img, (name,) in loader:
            img = img.to(device)
            probs = torch.sigmoid(model(img))
            pred = (probs > threshold).float()[0, 0].cpu().numpy()
            pred = remove_small_objects(pred, min_area)

            if pred.sum() == 0:
                empty += 1

            stem = Path(name).stem
            cv2.imwrite(str(output_dir / f"{stem}_pred.png"), (pred * 255).astype(np.uint8))
            if overlay_dir:
                overlay = make_overlay(img[0, 0].cpu().numpy(), pred)
                cv2.imwrite(str(overlay_dir / f"{stem}_overlay.png"), overlay)
            written += 1

    return written, empty


def predict_dataset(
    model: UNet,
    root: Path,
    device: torch.device,
    threshold: float,
    min_area: int,
    channel: str = "BF",
    save_overlay: bool = False,
) -> None:
    """Predict ``<session>/<channel>_circle`` for every session under ``root``."""
    input_dirs = sorted(p for p in root.glob(f"*/{channel}_circle") if p.is_dir())
    if not input_dirs:
        raise SystemExit(f"no */{channel}_circle directories found under {root}")

    total = 0
    total_empty = 0
    for input_dir in input_dirs:
        session = input_dir.parent
        written, empty = predict_directory(
            model,
            input_dir,
            session / "result_mask",
            device,
            threshold,
            min_area,
            session / "result_overlay" if save_overlay else None,
        )
        print(f"{session.name}: {written} masks ({empty} empty)")
        total += written
        total_empty += empty

    print(f"\n{total} masks written, {total_empty} empty.")
    if total_empty:
        print(
            "An empty mask means no predicted object survived the "
            f"--min-area {min_area} filter; those wells contribute no "
            "measurement downstream."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict cell masks with a trained U-Net."
    )
    parser.add_argument("--model", type=Path, required=True, help="checkpoint (.pth)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--root",
        type=Path,
        help="dataset root containing <session>/<channel>_circle directories",
    )
    group.add_argument("--input", type=Path, help="a single directory of well images")
    parser.add_argument("--output", type=Path, help="output directory (with --input)")
    parser.add_argument("--channel", default="BF")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help=f"discard predicted objects smaller than this, in pixels "
        f"(default: {DEFAULT_MIN_AREA})",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="also write red-tinted overlays for visual inspection",
    )
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}, min area: {args.min_area} px\n")

    model = load_model(args.model, device)

    if args.root:
        predict_dataset(
            model,
            args.root,
            device,
            args.threshold,
            args.min_area,
            args.channel,
            args.save_overlay,
        )
    else:
        if not args.output:
            parser.error("--output is required with --input")
        written, empty = predict_directory(
            model,
            args.input,
            args.output,
            device,
            args.threshold,
            args.min_area,
            args.output.with_name(args.output.name + "_overlay")
            if args.save_overlay
            else None,
        )
        print(f"{written} masks written, {empty} empty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Measure fluorescence intensity and cell area for every well, across all samples.

For each well the script pairs three things:

    <sample>_after/FL_split/       fluorescence image
    <sample>_after/result_mask/    predicted cell mask, after exposure
    <sample>_before/result_mask/   predicted cell mask, before exposure

and writes one row per well to a single CSV covering every sample::

    python fl_intensity.py --root data/raw --output results/fl_intensity.csv

Fluorescence is imaged only after exposure, so the before-exposure session
contributes cell area alone.

Sample directories are named ``<sample>_before`` / ``<sample>_after``, e.g.
``withDPP_sample_1_before``; the sample name is the part before the suffix.

Wells are matched on the position key embedded in every filename - region,
sub-position and well - so the channel prefix and the condition suffix do
not have to line up between directories.

Mean intensity is the mean over mask pixels of the green channel. The
fluorescence image is centre-cropped to the mask size, since the mask comes
from a 448 x 448 centre crop of the 450 x 450 well.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Tokens after "split" that identify a well: region, sub_x, sub_y and the
# two-part well suffix.
KEY_LENGTH = 5


# ---------------------------------------------------------------------------
# Filename handling
# ---------------------------------------------------------------------------


def well_key(stem: str) -> tuple[str, ...] | None:
    """Position key shared by every file for one well.

    ``FL_split_A_4_3_0_0``                     -> ('A', '4', '3', '0', '0')
    ``BF_split_A_4_3_0_0_24.0_0.5_pred``       -> ('A', '4', '3', '0', '0')

    Returns None if the name does not contain a ``split`` token followed by
    a full key.
    """
    parts = stem.split("_")
    if "split" not in parts:
        return None
    start = parts.index("split") + 1
    key = parts[start : start + KEY_LENGTH]
    return tuple(key) if len(key) == KEY_LENGTH else None


def condition_from_mask(stem: str) -> tuple[float, float] | None:
    """Laser power and duration from a mask filename.

    ``BF_split_A_4_3_0_0_24.0_0.5_pred`` -> (24.0, 0.5)
    """
    parts = stem.split("_")
    if parts[-1] == "pred":
        parts = parts[:-1]
    try:
        return float(parts[-2]), float(parts[-1])
    except (IndexError, ValueError):
        return None


def index_by_key(directory: Path) -> dict[tuple[str, ...], Path]:
    """Map each well key to its file in ``directory``."""
    index: dict[tuple[str, ...], Path] = {}
    if not directory.is_dir():
        return index
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = well_key(path.stem)
        if key is None:
            continue
        if key in index:
            print(
                f"warning: two files for well {key} in {directory.name}: "
                f"{index[key].name}, {path.name}",
                file=sys.stderr,
            )
            continue
        index[key] = path
    return index


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def read_mask(path: Path) -> np.ndarray | None:
    """Binary mask as float32 0/1."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else (img > 127).astype(np.float32)


def read_green_channel(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """Green channel of a fluorescence image, centre-cropped to ``shape``."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    green = img[:, :, 1] if img.ndim == 3 else img

    target_h, target_w = shape
    h, w = green.shape
    if (h, w) == (target_h, target_w):
        return green
    if h < target_h or w < target_w:
        return None
    top, left = (h - target_h) // 2, (w - target_w) // 2
    return green[top : top + target_h, left : left + target_w]


def mean_intensity(image: np.ndarray, mask: np.ndarray) -> float:
    """Mean of ``image`` over the mask. NaN for an empty mask."""
    total = mask.sum()
    return float((image * mask).sum() / total) if total else float("nan")


def mask_area(mask: np.ndarray) -> float:
    """Area of the largest connected component, in pixels.

    This is the contour area of the largest object, matching how the cell is
    delineated elsewhere in the pipeline; it is not the raw pixel count.
    """
    contours, _ = cv2.findContours(
        (mask > 0.5).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return float("nan")
    return float(cv2.contourArea(max(contours, key=cv2.contourArea)))


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------


def process_sample(sample: str, after_dir: Path, before_dir: Path) -> list[dict]:
    """One row per well that has an after mask, a before mask and an image."""
    after_masks = index_by_key(after_dir / "result_mask")
    before_masks = index_by_key(before_dir / "result_mask")
    after_fl = index_by_key(after_dir / "FL_split")

    if not after_masks:
        print(f"{sample}: no masks in {after_dir / 'result_mask'}", file=sys.stderr)
        return []

    rows: list[dict] = []
    missing_before = 0
    missing_image = 0

    for key, mask_path in after_masks.items():
        condition = condition_from_mask(mask_path.stem)
        if condition is None:
            print(f"{sample}: cannot read condition from {mask_path.name}", file=sys.stderr)
            continue
        power, duration = condition

        after_mask = read_mask(mask_path)
        if after_mask is None or after_mask.sum() == 0:
            continue

        before_path = before_masks.get(key)
        before_mask = read_mask(before_path) if before_path else None
        if before_mask is None or before_mask.sum() == 0:
            missing_before += 1
            continue

        image_path = after_fl.get(key)
        image = read_green_channel(image_path, after_mask.shape) if image_path else None
        if image is None:
            missing_image += 1
            continue

        rows.append(
            {
                "sample_name": sample,
                "filename": image_path.name,
                "power": power,
                "duration": duration,
                "after_fl_intensity": mean_intensity(image, after_mask),
                "after_area": mask_area(after_mask),
                "before_area": mask_area(before_mask),
            }
        )

    notes = []
    if missing_before:
        notes.append(f"{missing_before} without a usable before mask")
    if missing_image:
        notes.append(f"{missing_image} without a fluorescence image")
    suffix = f" ({', '.join(notes)})" if notes else ""
    print(f"{sample}: {len(rows)} wells{suffix}")
    return rows


def find_samples(root: Path) -> list[tuple[str, Path, Path]]:
    """Every ``<sample>_after`` directory paired with its ``_before``."""
    samples: list[tuple[str, Path, Path]] = []
    for after_dir in sorted(root.glob("*_after")):
        if not after_dir.is_dir():
            continue
        sample = after_dir.name[: -len("_after")]
        before_dir = root / f"{sample}_before"
        if not before_dir.is_dir():
            print(f"{sample}: no {before_dir.name}, skipping", file=sys.stderr)
            continue
        samples.append((sample, after_dir, before_dir))
    return samples


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

COLUMNS = [
    "sample_name",
    "filename",
    "power",
    "duration",
    "after_fl_intensity",
    "after_area",
    "before_area",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure fluorescence intensity and cell area for every well."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="dataset root containing <sample>_before / <sample>_after directories",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="output CSV covering all samples"
    )
    args = parser.parse_args(argv)

    samples = find_samples(args.root)
    if not samples:
        raise SystemExit(f"no *_after directories found under {args.root}")

    rows: list[dict] = []
    for sample, after_dir, before_dir in samples:
        rows += process_sample(sample, after_dir, before_dir)

    if not rows:
        raise SystemExit("no wells measured")

    frame = pd.DataFrame(rows, columns=COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"\n{len(frame)} wells from {frame['sample_name'].nunique()} samples")
    print(f"conditions: {sorted(set(zip(frame['power'], frame['duration'])))}")
    print(f"CSV: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Well geometry
# ---------------------------------------------------------------------------

CROP_SIZE = 450  # px, square
JPEG_QUALITY = 100

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class Well:
    """One well within a raw field. ``x``/``y`` are top-left, in pixels."""

    suffix: str
    x: int
    y: int


# Most regions contain a single well in the field of view; region Y contains four. Keyed by region letter, with "*" as the default.
WELL_LAYOUTS: dict[str, list[Well]] = {
    "*": [
        Well("0_0", 758, 580),
    ],
    "Y": [
        Well("0_0", 758, 580),
        Well("0_1", 260, 580),
        Well("0_2", 260, 1085),
        Well("0_3", 758, 1085),
    ],
}


# ---------------------------------------------------------------------------
# Filename handling
# ---------------------------------------------------------------------------


def parse_stem(stem: str) -> tuple[str, str, str, str]:
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"unrecognised field name: {stem!r}")
    prefix = "_".join(parts[:-3])
    region, sub_x, sub_y = parts[-3:]
    return prefix, region, sub_x, sub_y


def wells_for(region: str) -> list[Well]:
    return WELL_LAYOUTS.get(region.upper(), WELL_LAYOUTS["*"])


def output_name(prefix: str, region: str, sub_x: str, sub_y: str, well: Well) -> str:
    """``BF`` + ``Y_0_1`` + well ``0_0`` -> ``BF_split_Y_0_1_0_0.jpg``."""
    return f"{prefix}_split_{region}_{sub_x}_{sub_y}_{well.suffix}.jpg"


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def crop_directory(
    input_dir: Path,
    output_dir: Path,
    crop_size: int = CROP_SIZE,
    quality: int = JPEG_QUALITY,
) -> int:
    """Crop every field in ``input_dir``. Returns the number of wells written."""
    fields = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not fields:
        print(f"  no images in {input_dir}", file=sys.stderr)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for field_path in fields:
        prefix, region, sub_x, sub_y = parse_stem(field_path.stem)
        with Image.open(field_path) as image:
            width, height = image.size
            for well in wells_for(region):
                if well.x + crop_size > width or well.y + crop_size > height:
                    print(
                        f"  skipped {field_path.name} well {well.suffix}: "
                        f"crop exceeds image bounds ({width}x{height})",
                        file=sys.stderr,
                    )
                    continue
                tile = image.crop(
                    (well.x, well.y, well.x + crop_size, well.y + crop_size)
                ).transpose(Image.FLIP_LEFT_RIGHT)
                tile.save(
                    output_dir / output_name(prefix, region, sub_x, sub_y, well),
                    "JPEG",
                    quality=quality,
                )
                written += 1

    print(f"  {input_dir.name}: {len(fields)} fields -> {written} wells")
    return written


def find_source_dirs(root: Path) -> list[Path]:
    """Every ``*_original`` directory under ``root``, in sorted order."""
    return sorted(p for p in root.glob("*/*_original") if p.is_dir())


def crop_dataset(root: Path) -> None:
    """Crop every ``*_original`` directory in the dataset tree."""
    source_dirs = find_source_dirs(root)
    if not source_dirs:
        raise SystemExit(f"no */*_original directories found under {root}")

    total = 0
    for source_dir in source_dirs:
        # BF_original -> BF_split, alongside the source
        target = source_dir.with_name(source_dir.name.replace("_original", "_split"))
        print(f"{source_dir.parent.name}/{source_dir.name}")
        total += crop_directory(source_dir, target)

    print(f"\n{len(source_dirs)} directories, {total} wells written.")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Crop well regions out of raw microscope fields."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--root",
        type=Path,
        help="dataset root containing <session>/<channel>_original directories",
    )
    group.add_argument("--input", type=Path, help="a single directory of raw fields")
    parser.add_argument(
        "--output", type=Path, help="output directory (required with --input)"
    )
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY)
    args = parser.parse_args(argv)

    if args.root:
        crop_dataset(args.root)
    else:
        if not args.output:
            parser.error("--output is required with --input")
        crop_directory(args.input, args.output, args.crop_size, args.quality)
    return 0


if __name__ == "__main__":
    sys.exit(main())
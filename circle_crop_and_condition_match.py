"""Mask each well image to its circular boundary and attach its laser condition.

For every cropped well image, the well wall is located with a Hough circle
transform and everything outside it is set to zero, so that downstream
segmentation sees only the well interior.

The output filename carries the laser condition, taken from the matching
entry in the ``*_rename`` directory produced during acquisition::

    9_after/BF_split/BF_split_A_0_0_0_0.jpg
      + 9_before/BF_split_rename/BF_split_A_0_0_0_0_5_1.0.jpg
      -> 9_after/BF_circle/BF_split_A_0_0_0_0_5_1.0.jpg

Conditions are assigned once per experiment, before exposure, so both the
pre- and post-exposure sessions read the same table from the ``_before``
directory. There is no need to copy it.

Run over the whole dataset::

    python circle_crop.py --root data/raw

or over a single session::

    python circle_crop.py --input  data/raw/9_after/BF_split \
                          --rename data/raw/9_after/BF_split_rename \
                          --output data/raw/9_after/BF_circle

Only exposed wells appear in the rename directory; a well with no entry was
never exposed and is written with the control suffix ``_0_0``. The number of
control wells is reported per session, so an unexpected count is visible
rather than silent. Wells whose circle is not detected are left unmasked and
listed at the end of the run. See ``--help`` for the Hough parameters.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Number of condition fields appended to the stem by the acquisition step
# (power and duration).
N_CONDITION_FIELDS = 2

# Only exposed wells are listed in the rename directory. A well with no entry
# there was never exposed, and is labelled as an unexposed control: zero
# power, zero duration.
CONTROL_LABEL = "0_0"


@dataclass(frozen=True)
class HoughParams:
    """Parameters of the Hough circle transform used to find the well wall.

    Tuned for 450 x 450 px wells at the magnification used in this study.
    """

    dp: float = 1.2
    min_dist: float = 600
    param1: float = 50
    param2: float = 30
    min_radius: int = 150
    max_radius: int = 300
    blur_kernel: int = 5
    # The detected radius is shrunk by this many pixels so the well wall
    # itself is excluded from the masked region.
    radius_margin: int = 15


# ---------------------------------------------------------------------------
# Condition matching
# ---------------------------------------------------------------------------


def build_condition_map(rename_dir: Path) -> dict[str, str]:
    """Map each well stem to its condition-tagged filename.

    ``BF_split_A_0_0_0_0_5_1.0.jpg`` is keyed by ``BF_split_A_0_0_0_0``.

    Raises if two condition files map to the same well, which would mean the
    assignment step ran more than once into the same directory.
    """
    mapping: dict[str, str] = {}
    for path in sorted(rename_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = path.stem.split("_")
        if len(parts) <= N_CONDITION_FIELDS:
            raise ValueError(f"unexpected condition filename: {path.name!r}")
        key = "_".join(parts[:-N_CONDITION_FIELDS])
        if key in mapping:
            raise ValueError(
                f"two condition files match well {key!r}: "
                f"{mapping[key]!r} and {path.name!r}"
            )
        mapping[key] = path.name
    return mapping


# ---------------------------------------------------------------------------
# Circular masking
# ---------------------------------------------------------------------------


def mask_to_circle(image: np.ndarray, params: HoughParams) -> tuple[np.ndarray, bool]:
    """Zero everything outside the detected well.

    Returns the masked image and whether a circle was found. If no circle is
    found the image is returned unchanged.
    """
    blurred = cv2.medianBlur(image, params.blur_kernel)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=params.dp,
        minDist=params.min_dist,
        param1=params.param1,
        param2=params.param2,
        minRadius=params.min_radius,
        maxRadius=params.max_radius,
    )
    if circles is None:
        return image, False

    # Strongest candidate only; minDist is large enough that one well
    # yields one circle.
    x, y, radius = np.around(circles[0][0]).astype(int)
    radius = max(radius - params.radius_margin, 1)

    height, width = image.shape
    grid_y, grid_x = np.ogrid[:height, :width]
    inside = (grid_x - x) ** 2 + (grid_y - y) ** 2 <= radius**2

    masked = image.copy()
    masked[~inside] = 0
    return masked, True


# ---------------------------------------------------------------------------
# Per-directory processing
# ---------------------------------------------------------------------------


def process_directory(
    input_dir: Path,
    output_dir: Path,
    rename_dir: Path | None,
    params: HoughParams,
    control_label: str = CONTROL_LABEL,
) -> tuple[int, int, list[str]]:
    """Mask every well in ``input_dir``.

    Returns (number written, number labelled as control, wells with no circle).
    """
    wells = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not wells:
        print(f"  no images in {input_dir}", file=sys.stderr)
        return 0, 0, []

    conditions = build_condition_map(rename_dir) if rename_dir else {}
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    controls = 0
    undetected: list[str] = []

    for well_path in wells:
        image = cv2.imread(str(well_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  unreadable: {well_path.name}", file=sys.stderr)
            continue

        if rename_dir:
            output_name = conditions.get(well_path.stem)
            if output_name is None:
                # Only exposed wells appear in the rename directory, so a
                # well with no entry is an unexposed control.
                output_name = f"{well_path.stem}_{control_label}{well_path.suffix}"
                controls += 1
        else:
            output_name = well_path.name

        masked, detected = mask_to_circle(image, params)
        if not detected:
            undetected.append(well_path.name)

        cv2.imwrite(str(output_dir / output_name), masked)
        written += 1

    return written, controls, undetected


# ---------------------------------------------------------------------------
# Dataset traversal
# ---------------------------------------------------------------------------


def rename_dir_for(session: Path, channel: str) -> Path:
    """Locate the condition-tagged filenames for ``session``.

    Laser conditions are assigned once per experiment, before exposure, so
    the post-exposure sessions read the table from their paired ``_before``
    session rather than holding a copy of it::

        9_after  ->  9_before/BF_split_rename

    Pre- and post-exposure wells share the same stem, so the mapping applies
    unchanged.
    """
    if session.name.endswith("_before"):
        return session / f"{channel}_split_rename"
    prefix = session.name.rsplit("_", 1)[0]
    return session.parent / f"{prefix}_before" / f"{channel}_split_rename"


def process_dataset(
    root: Path,
    params: HoughParams,
    channel: str = "BF",
) -> None:
    """Process ``<session>/<channel>_split`` for every session under ``root``."""
    input_dirs = sorted(p for p in root.glob(f"*/{channel}_split") if p.is_dir())
    if not input_dirs:
        raise SystemExit(f"no */{channel}_split directories found under {root}")

    total = 0
    total_controls = 0
    all_undetected: list[str] = []

    for input_dir in input_dirs:
        session = input_dir.parent
        rename_dir = rename_dir_for(session, channel)
        output_dir = session / f"{channel}_circle"

        if not rename_dir.is_dir():
            print(
                f"{session.name}: condition directory not found at {rename_dir}, "
                "skipping",
                file=sys.stderr,
            )
            continue

        written, controls, undetected = process_directory(
            input_dir, output_dir, rename_dir, params
        )
        print(
            f"{session.name}: {written} wells -> {output_dir.name} "
            f"({written - controls} exposed, {controls} control)"
        )
        total += written
        total_controls += controls
        all_undetected += [f"{session.name}/{n}" for n in undetected]

    print(f"\n{total} wells written ({total - total_controls} exposed, "
          f"{total_controls} control).")
    report_problems(all_undetected)


def report_problems(undetected: list[str]) -> None:
    """Print the wells that need attention before the next pipeline step."""
    if undetected:
        print(
            f"\n{len(undetected)} well(s) with no circle detected "
            "(saved unmasked - the whole field will be measured):"
        )
        for name in undetected:
            print(f"  {name}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mask well images to their circular boundary and attach "
        "the laser condition to each filename."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--root",
        type=Path,
        help="dataset root containing <session>/<channel>_split directories",
    )
    group.add_argument("--input", type=Path, help="a single directory of well images")
    parser.add_argument("--output", type=Path, help="output directory (with --input)")
    parser.add_argument(
        "--rename",
        type=Path,
        help="directory of condition-tagged filenames (with --input); "
        "omit to keep the input filenames",
    )
    parser.add_argument(
        "--channel", default="BF", help="channel prefix to process (default: BF)"
    )
    parser.add_argument(
        "--control-label",
        default=CONTROL_LABEL,
        help="condition suffix for wells absent from the rename directory, "
        f"i.e. unexposed controls (default: {CONTROL_LABEL})",
    )

    hough = parser.add_argument_group("Hough circle parameters")
    defaults = HoughParams()
    hough.add_argument("--dp", type=float, default=defaults.dp)
    hough.add_argument("--min-dist", type=float, default=defaults.min_dist)
    hough.add_argument("--param1", type=float, default=defaults.param1)
    hough.add_argument("--param2", type=float, default=defaults.param2)
    hough.add_argument("--min-radius", type=int, default=defaults.min_radius)
    hough.add_argument("--max-radius", type=int, default=defaults.max_radius)
    hough.add_argument("--blur-kernel", type=int, default=defaults.blur_kernel)
    hough.add_argument(
        "--radius-margin",
        type=int,
        default=defaults.radius_margin,
        help="pixels to shrink the detected radius by, excluding the well wall",
    )

    args = parser.parse_args(argv)
    params = HoughParams(
        dp=args.dp,
        min_dist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        blur_kernel=args.blur_kernel,
        radius_margin=args.radius_margin,
    )

    if args.root:
        process_dataset(args.root, params, args.channel)
    else:
        if not args.output:
            parser.error("--output is required with --input")
        written, controls, undetected = process_directory(
            args.input, args.output, args.rename, params, args.control_label
        )
        print(f"{written} wells written ({written - controls} exposed, "
              f"{controls} control).")
        report_problems(undetected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
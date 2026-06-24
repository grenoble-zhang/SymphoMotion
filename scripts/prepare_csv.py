#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


REQUIRED_SAMPLE_FILES = (
    "first_image.png",
    "full_prompt.json",
    "spatialtracker2.npz",
    "render_output/render_with_2d_bbox.mp4",
    "render_output/render_mask.mp4",
)


def is_sample_dir(path: Path, require_gt_video: bool) -> bool:
    if not path.is_dir():
        return False
    for rel_path in REQUIRED_SAMPLE_FILES:
        if not (path / rel_path).is_file():
            return False
    if require_gt_video and not (path / f"{path.name}.mp4").is_file():
        return False
    return True


def find_samples(root: Path, require_gt_video: bool):
    candidates = {p.parent for p in root.rglob("first_image.png")}
    samples = [p for p in candidates if is_sample_dir(p, require_gt_video)]
    return sorted(samples)


def main():
    parser = argparse.ArgumentParser(description="Build a SymphoMotion CSV manifest.")
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--no_require_gt_video",
        action="store_true",
        help="Allow samples without the original <sample_name>.mp4 video.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples = find_samples(root, require_gt_video=not args.no_require_gt_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"path": str(sample)})

    print(f"Wrote {len(samples)} samples to {output}")


if __name__ == "__main__":
    main()

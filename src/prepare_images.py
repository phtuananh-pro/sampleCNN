"""Create and reuse a resized image cache for the CNN training dataset."""

import argparse
import json
from pathlib import Path

from PIL import Image

try:  # Works both with ``python -m src.prepare_images`` and a file path.
    from .dataset import IMAGE_EXTENSIONS
    from .logging_utils import log
except ImportError:  # pragma: no cover - only used for direct file execution.
    from dataset import IMAGE_EXTENSIONS
    from logging_utils import log


def source_images(data_root: Path):
    """Yield supported images from train/ and val/ while preserving layout."""
    for split in ("train", "val"):
        split_root = data_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Expected dataset split folder: {split_root}")

        for path in sorted(split_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def load_manifest(path: Path) -> dict:
    """Read the prior cache metadata, returning an empty manifest if invalid."""
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("version") == 1:
            return manifest
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def prepare_images(data_root: Path, output_root: Path, image_size: int) -> None:
    """Resize new or changed source files and reuse valid cached images."""
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    if output_root == data_root or output_root.is_relative_to(data_root):
        raise ValueError("--output must be outside the source dataset folder")
    if image_size <= 0:
        raise ValueError("--image-size must be a positive integer")

    manifest_path = output_root / ".resize_cache.json"
    previous = load_manifest(manifest_path)
    old_entries = previous.get("files", {})
    new_entries = {}
    resized = reused = 0
    resampling = getattr(Image, "Resampling", Image)

    for source in source_images(data_root):
        relative = source.relative_to(data_root)
        relative_key = relative.as_posix()
        destination = output_root / relative
        source_stat = source.stat()
        old = old_entries.get(relative_key, {})
        unchanged = (
            old.get("source_size") == source_stat.st_size
            and old.get("source_mtime_ns") == source_stat.st_mtime_ns
            and old.get("image_size") == image_size
            and destination.is_file()
        )

        if unchanged:
            reused += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image = image.convert("RGB")
                if image.size != (image_size, image_size):
                    image = image.resize((image_size, image_size), resampling.BILINEAR)
                image.save(destination)
            resized += 1

        new_entries[relative_key] = {
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "image_size": image_size,
        }

    # Remove stale outputs only when the previous manifest says this script
    # created them. Other files in the output directory are left untouched.
    current_keys = set(new_entries)
    for stale_key, old in old_entries.items():
        if stale_key in current_keys:
            continue
        stale_output = output_root / Path(stale_key)
        if stale_output.is_file():
            stale_output.unlink()

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".tmp")
    with temporary_manifest.open("w", encoding="utf-8") as file:
        json.dump(
            {"version": 1, "image_size": image_size, "files": new_entries},
            file,
            indent=2,
        )
    temporary_manifest.replace(manifest_path)

    if not new_entries:
        raise ValueError(f"No supported images found under {data_root / 'train'} and {data_root / 'val'}")

    log(f"resized={resized} reused={reused} image_size={image_size}")
    log(f"ready dataset={output_root}")


def parse_args():
    """Define the source dataset, output cache, and target image size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cats_dogs", help="source folder containing train/ and val/")
    parser.add_argument(
        "--output",
        default=None,
        help="cache folder (default: sibling folder named <data>_resized_<image-size>)",
    )
    parser.add_argument("--image-size", type=int, default=64, help="square output image size")
    return parser.parse_args()


def main(args):
    """Resolve the default output directory and build or reuse its cache."""
    data_root = Path(args.data)
    output_root = (
        Path(args.output)
        if args.output
        else data_root.with_name(f"{data_root.name}_resized_{args.image_size}")
    )
    prepare_images(data_root, output_root, args.image_size)


if __name__ == "__main__":
    main(parse_args())

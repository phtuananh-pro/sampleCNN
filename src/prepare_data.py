"""Make a small train/validation dataset from raw class folders.

Many public datasets are downloaded in a form like:

    raw/PetImages/Cat/*.jpg
    raw/PetImages/Dog/*.jpg

The training code expects a slightly different but very common layout:

    data/cats_dogs/train/cat/*.jpg
    data/cats_dogs/train/dog/*.jpg
    data/cats_dogs/val/cat/*.jpg
    data/cats_dogs/val/dog/*.jpg

This helper validates files, takes a deterministic small subset, and copies it
into that layout.  Keeping the split explicit makes the relationship between
training data and validation data easy to inspect while learning.
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image

try:  # Works both with ``python -m src.prepare_data`` and a file path.
    from .dataset import IMAGE_EXTENSIONS
except ImportError:  # pragma: no cover - only used for direct file execution.
    from dataset import IMAGE_EXTENSIONS


def readable_images(folder: Path):
    """Yield supported image files that Pillow can actually decode.

    Some image archives contain damaged files or files with an image suffix
    that are not readable.  ``Image.verify`` checks the file without fully
    decoding it into a training tensor, so those files can be skipped before
    the dataset is copied.
    """
    for path in sorted(folder.rglob("*")):
        # Ignore directories and file types the CNN loader does not support.
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            # A bad file should not prevent a small educational example from
            # being built.  It is simply not included in the returned subset.
            continue

        yield path


def main(args):
    """Validate, split, and copy a bounded number of images per class."""
    source = Path(args.source)
    output = Path(args.output)

    # Every immediate directory under source is interpreted as one class.  The
    # sorted order makes the selection deterministic across runs.
    class_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found in {source}")

    for class_dir in class_dirs:
        # Sorting plus slicing creates a reproducible, intentionally small
        # experiment.  The helper does not download or transform the images;
        # dataset.py performs the resize when training or inference starts.
        images = list(readable_images(class_dir))[: args.per_class]

        # We need at least one image left for training after reserving the
        # requested validation images.  The strict check catches a typo in the
        # command line early rather than creating an empty split.
        if len(images) <= args.val_count:
            raise ValueError(f"Need more than --val-count readable images in {class_dir}")

        # The first images become validation data; the remainder become
        # training data.  For a real benchmark, a randomized or stratified
        # split may be preferable, but a deterministic split is clearer here.
        split_selections = (("val", images[: args.val_count]), ("train", images[args.val_count :]))
        for split, selected in split_selections:
            destination = output / split / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)

            for index, image in enumerate(selected):
                # Prefixing with an index avoids collisions if two source
                # folders contain identically named files.  Keep the original
                # suffix because Pillow and the loader recognize it.
                shutil.copy2(image, destination / f"{index:04d}{image.suffix.lower()}")

        print(f"{class_dir.name}: {len(images) - args.val_count} train, {args.val_count} val")


def parse_args():
    """Define the source, destination, and small-subset sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="folder containing one folder per class")
    parser.add_argument("--output", default="data/cats_dogs")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

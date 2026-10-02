"""A tiny image-folder dataset using only Pillow and PyTorch.

This module keeps the data pipeline explicit so a learner can connect it to
the chapter's tensor notation.  An image starts as a Pillow image with height,
width, and RGB values.  It becomes a PyTorch tensor with the channel-first
shape expected by ``nn.Conv2d``:

    one image:  (3, H, W)
    one batch:  (B, 3, H, W)

The loader does not use ``torchvision``.  That keeps the dependency list short
and makes resizing, normalization, and the small training augmentation visible.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


# These are the file types that the example will consider images.  The dataset
# does not need to know anything about the class names: every immediate folder
# under ``root`` is treated as one class, such as ``cat`` or ``dog``.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """Convert one Pillow image into a normalized ``(3, image_size, image_size)`` tensor.

    This function is the bridge between ordinary image files and the CNN input
    described in the chapter:

    1. ``convert("RGB")`` guarantees exactly three input channels.
    2. ``resize`` gives every image the same spatial dimensions, which lets
       images be stacked into a batch.
    3. Pixel values are copied into a float tensor and rearranged from Pillow's
       ``(H, W, C)`` convention to PyTorch's ``(C, H, W)`` convention.
    4. Division by 255 changes 8-bit values in ``[0, 255]`` to approximately
       ``[0, 1]``.  The CNN then receives numerical inputs on a small scale.
    """
    # Pillow changed the location of the resampling constants in newer
    # versions.  This fallback keeps the example compatible with both forms.
    resampling = getattr(Image, "Resampling", Image)

    # RGB is important even if an input file is grayscale or has transparency:
    # the first convolution layer is defined with 3 input channels.
    image = image.convert("RGB").resize((image_size, image_size), resampling.BILINEAR)

    # ``getdata`` returns one (R, G, B) tuple per pixel in row-major order.
    # Creating a float tensor here makes the later division and model input
    # type explicit.  A production pipeline might use a faster conversion,
    # but this version is easier to read for a learner.
    pixels = torch.tensor(list(image.getdata()), dtype=torch.float32)

    # At this point the shape is (H, W, 3), matching the way image libraries
    # commonly expose pixels.  Conv2d expects (C, H, W), so move the channel
    # dimension to the front.  This is exactly the C x H x W notation in the
    # chapter; a DataLoader will add B at the front later.
    pixels = pixels.reshape(image_size, image_size, 3).permute(2, 0, 1)

    # Normalize 8-bit pixels.  The tensor now has shape (3, image_size,
    # image_size) and is ready for a convolutional model.
    return pixels / 255.0


class ImageFolderDataset(Dataset):
    """Read images from ``root/class_name/image`` without requiring torchvision.

    The expected layout is:

        root/
        ├── cat/
        │   ├── example1.jpg
        │   └── example2.jpg
        └── dog/
            └── example3.jpg

    Class names are sorted so the mapping is deterministic.  In the repository
    this normally means ``cat -> 0`` and ``dog -> 1``.  The integer labels are
    what ``CrossEntropyLoss`` expects during training.

    Parameters
    ----------
    root:
        Directory whose child directories are class names.
    image_size:
        Target height and width for every returned image.
    training:
        If true, randomly flip half of the images horizontally.  This is a
        small augmentation, not a new dependency.  It is disabled for
        validation and inference so their results are repeatable.
    """

    def __init__(self, root: str | Path, image_size: int = 64, training: bool = False):
        # Convert strings from command-line arguments into Path objects once.
        self.root = Path(root)
        self.image_size = image_size
        self.training = training

        # Each subdirectory is one class.  Sorting makes the class-to-index
        # mapping stable across runs and across train/validation datasets.
        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not self.classes:
            raise ValueError(f"No class folders found in {self.root}")

        # The model works with integer labels, while humans think in names.
        # Keep both representations: class_to_idx is used by samples, and
        # classes is saved in the checkpoint for readable inference output.
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        # Build an index of (image_path, integer_label) pairs up front.  Doing
        # this once in __init__ makes __getitem__ simple and lets __len__ tell
        # the DataLoader how many examples exist.
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            # rglob also accepts images in nested folders, while sorted keeps
            # the order deterministic for debugging and reproducibility.
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))
        if not self.samples:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        """Return the number of indexed images."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Load one sample and return ``(image_tensor, integer_label)``."""
        # A Dataset returns one example at a time.  DataLoader calls this method
        # repeatedly and stacks the returned tensors into (B, C, H, W).
        path, label = self.samples[index]

        # Copy the decoded image before leaving the context manager.  This
        # closes the file promptly and leaves an in-memory image for conversion.
        with Image.open(path) as image:
            image = image.copy()

        tensor = image_to_tensor(image, self.image_size)

        # Horizontal reflection is a simple way to show data augmentation.  It
        # creates a slightly different view of the same class without changing
        # the label.  ``dims=(2,)`` flips the width axis of (C, H, W).
        if self.training and torch.rand(()) < 0.5:
            tensor = torch.flip(tensor, dims=(2,))

        return tensor, label


def load_image(path: str | Path, image_size: int) -> torch.Tensor:
    """Load one image as a batched tensor for inference.

    The dataset returns an unbatched ``(C, H, W)`` tensor because DataLoader
    adds the batch dimension.  Inference has only one image and no DataLoader,
    so ``unsqueeze(0)`` explicitly creates ``(1, C, H, W)`` for the CNN.
    """
    with Image.open(path) as image:
        tensor = image_to_tensor(image.copy(), image_size)
    return tensor.unsqueeze(0)

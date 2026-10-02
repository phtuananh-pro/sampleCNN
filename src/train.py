"""Train the small CNN on ``data/train`` and ``data/val`` image folders.

The training script follows the chapter's complete learning loop:

    forward:  image -> CNN -> logits -> loss
    backward: loss -> classifier -> convolutions -> parameter gradients
    update:   optimizer uses gradients to change the shared kernels

It intentionally avoids a training framework so each step remains visible.
Run it from the repository root with ``python -m src.train --help``.
"""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:  # Works both with ``python -m src.train`` and ``python src/train.py``.
    from .cnn import SimpleCNN
    from .dataset import ImageFolderDataset
except ImportError:  # pragma: no cover - only used for direct file execution.
    from cnn import SimpleCNN
    from dataset import ImageFolderDataset


def set_seed(seed: int) -> None:
    """Make the small example's random choices repeatable where possible."""
    # Python's random module controls ordinary Python randomness.  The dataset
    # uses torch.rand for horizontal flips, and model initialization also uses
    # PyTorch's random generator, so seed both sources.
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    """Measure average loss and accuracy without changing model parameters."""
    # Evaluation must not update weights.  ``eval`` changes layers that have
    # training-specific behavior (none are essential in this tiny model, but it
    # is the correct habit), and no_grad avoids storing backward activations.
    model.eval()
    correct = total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        # DataLoader gives images as (B, 3, H, W) and labels as (B,).  Moving
        # both to the same device is required before the forward computation.
        images, labels = images.to(device), labels.to(device)

        # This context is intentionally inside evaluate rather than the caller:
        # validation only needs predictions, not gradients for an update.
        with torch.no_grad():
            logits = model(images)

        # CrossEntropyLoss compares logits with integer class labels.  The
        # multiplication by batch size lets us compute a true dataset average
        # even when the last batch is smaller than the others.
        batch_loss = criterion(logits, labels)
        loss_total += batch_loss.item() * labels.size(0)

        # argmax selects the class with the largest logit.  Softmax is not
        # needed for choosing the class because it preserves the ordering.
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return loss_total / total, correct / total


def train(args):
    """Build the data pipeline and run the forward/backward/update loop."""
    # Reproducibility makes it easier for a learner to connect a code change to
    # a changed result.  Exact bit-for-bit reproducibility can still depend on
    # hardware and PyTorch backend details.
    set_seed(args.seed)

    # Use a GPU if one is available unless the learner explicitly requests
    # CPU.  The model and each batch must live on the same device.
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_root = Path(args.data)

    # Each dataset reads root/class_name/image.  The training dataset may flip
    # images; validation never does.  Both use the same image size so their
    # tensors have compatible shapes.
    train_set = ImageFolderDataset(data_root / "train", args.image_size, training=True)
    val_set = ImageFolderDataset(data_root / "val", args.image_size)
    if train_set.classes != val_set.classes:
        raise ValueError("train and val must contain the same class folders")

    # DataLoader turns individual samples into batches.  ``num_workers=0`` is
    # deliberate: it is simple, friendly to Windows, and adequate for this
    # small educational dataset.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # The number of output logits is determined by the number of class folders.
    # For cats and dogs, SimpleCNN ends with two logits per image.
    model = SimpleCNN(len(train_set.classes)).to(device)

    # CrossEntropyLoss combines a log-softmax with negative log likelihood. It
    # expects raw logits from the model and integer labels such as cat=0/dog=1.
    criterion = nn.CrossEntropyLoss()

    # SGD is intentionally chosen because it makes the chapter's update rule
    # easy to recognize: parameter <- parameter - learning_rate * gradient.
    # Momentum, schedulers, and other optimization improvements are omitted.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    print(f"device={device} classes={train_set.classes} train={len(train_set)} val={len(val_set)}")
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        # ``train`` enables training behavior in the model and its children.
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # ------------------------- forward pass ----------------------
            # The CNN produces one logit vector per image.  The shape changes
            # inside SimpleCNN from (B, 3, 64, 64) to (B, num_classes).
            logits = model(images)
            loss = criterion(logits, labels)

            # ------------------------- backward pass ---------------------
            # Clear gradients left by the previous batch.  PyTorch accumulates
            # gradients by default, which is useful sometimes but not here.
            optimizer.zero_grad()

            # Differentiate the loss through the linear head, global average
            # pooling, ReLUs, max pools, and convolutions.  For a shared kernel,
            # gradient contributions from all spatial uses accumulate on the
            # same kernel parameters, exactly as described in the chapter.
            loss.backward()

            # ------------------------- update step -----------------------
            # Adjust every trainable parameter using its newly computed
            # gradient, including all shared convolution kernels and biases.
            optimizer.step()

            # Keep a sample-count-weighted total for a meaningful epoch loss.
            running_loss += loss.item() * labels.size(0)

        # Validation is a separate forward-only pass over data not used for
        # parameter updates.  It gives a simple signal about generalization.
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        train_loss = running_loss / len(train_set)
        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.1%}"
        )

        # Save the best validation model rather than only the final epoch.  The
        # class names and image size travel with the weights so infer.py can
        # reconstruct the same output mapping and preprocessing later.
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            checkpoint_path = Path(args.checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_set.classes,
                    "image_size": args.image_size,
                },
                checkpoint_path,
            )

    print(f"saved best checkpoint to {args.checkpoint} ({best_accuracy:.1%})")


def parse_args():
    """Define the few command-line knobs needed for a small experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cats_dogs", help="folder containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--checkpoint", default="models/cnn_cats_dogs.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true", help="force CPU even when CUDA is available")
    return parser.parse_args()


if __name__ == "__main__":
    # Keeping this guard means importing train.py does not start training.  A
    # command-line invocation parses arguments and then enters the loop above.
    train(parse_args())

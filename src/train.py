"""Train the small CNN on ``data/train`` and ``data/val`` image folders.

The training script follows the chapter's complete learning loop:

    forward:  image -> CNN -> logits -> loss
    backward: loss -> classifier -> convolutions -> parameter gradients
    update:   optimizer uses gradients to change the shared kernels

It intentionally avoids a training framework so each step remains visible.
The learning loop illustrates Chapter VI.3, Sections 15-17. Run it from the
repository root with ``python -m src.train --help``.
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
    from .logging_utils import log, log_model_info
except ImportError:  # pragma: no cover - only used for direct file execution.
    from cnn import SimpleCNN
    from dataset import ImageFolderDataset
    from logging_utils import log, log_model_info


def set_seed(seed: int) -> None:
    """Make the small example's random choices repeatable where possible."""
    # Python's random module controls ordinary Python randomness.  The dataset
    # uses torch.rand for horizontal flips, and model initialization also uses
    # PyTorch's random generator, so seed both sources.
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device, num_classes):
    """Measure loss, accuracy, and class-wise metrics on a labeled split."""
    # Evaluation must not update weights.  ``eval`` changes layers that have
    # training-specific behavior (none are essential in this tiny model, but it
    # is the correct habit), and no_grad avoids storing backward activations.
    model.eval()
    correct = total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

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
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        encoded_pairs = labels * num_classes + predictions
        confusion += torch.bincount(encoded_pairs, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )

    true_positives = confusion.diag().to(torch.float32)
    precision = true_positives / confusion.sum(dim=0).clamp_min(1)
    recall = true_positives / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return loss_total / total, correct / total, confusion, precision, recall, f1


def train(args):
    """Build the data pipeline and run the forward/backward/update loop."""
    # Reproducibility makes it easier for a learner to connect a code change to
    # a changed result.  Exact bit-for-bit reproducibility can still depend on
    # hardware and PyTorch backend details.
    set_seed(args.seed)

    # ``auto`` selects CUDA when this PyTorch installation can access it.  A
    # CUDA-capable GPU alone is not enough: PyTorch must also be a CUDA build.
    # The model and each batch must live on the same device.
    if args.device == "cuda" and not torch.cuda.is_available():
        cuda_build = torch.version.cuda or "not installed (CPU-only PyTorch)"
        raise RuntimeError(
            "--device cuda was requested, but PyTorch cannot access CUDA. "
            f"torch={torch.__version__}, torch CUDA build={cuda_build}. "
            "Install a CUDA-enabled PyTorch build and check that the NVIDIA "
            "driver is up to date."
        )

    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
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

    # The optimizer changes how gradients update parameters; the same
    # forward/loss/backward loop works with SGD, Adam, or AdamW.
    optimizer_classes = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    optimizer_class = optimizer_classes[args.optimizer]
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)

    log(
        f"optimizer={optimizer_class.__name__} learning_rate={args.learning_rate:g} "
        f"epochs={args.epochs} batch_size={args.batch_size} image_size={args.image_size}"
    )
    if device.type == "cuda":
        log(f"device={device} gpu={torch.cuda.get_device_name(device)}")
    else:
        log(f"device={device}")
    log(f"classes={train_set.classes} train={len(train_set)} val={len(val_set)}")
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        # Report progress before processing this epoch so long runs always
        # show which epoch is currently underway.
        log(f"epoch {epoch:02d}/{args.epochs} starting")

        # ``train`` enables training behavior in the model and its children.
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # ------------------------- forward pass ----------------------
            # The CNN produces one logit vector per image. With the default
            # 64-pixel input, its shape changes from (B, 3, 64, 64) to
            # (B, num_classes); --image-size changes the input height and width.
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
        val_loss, val_accuracy, confusion, precision, recall, f1 = evaluate(
            model, val_loader, device, len(train_set.classes)
        )
        train_loss = running_loss / len(train_set)
        log(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.1%} val_macro_f1={f1.mean().item():.1%}"
        )
        for index, class_name in enumerate(train_set.classes):
            log(
                f"val_class={class_name} precision={precision[index].item():.1%} "
                f"recall={recall[index].item():.1%} f1={f1[index].item():.1%}"
            )
        log(
            f"val_confusion_matrix rows=actual columns=predicted "
            f"classes={train_set.classes} values={confusion.cpu().tolist()}"
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
                    "optimizer": args.optimizer,
                    "learning_rate": args.learning_rate,
                },
                checkpoint_path,
            )

    log(f"training complete; best validation accuracy={best_accuracy:.1%}")
    log_model_info(model, train_set.classes, args.image_size, device, args.checkpoint)
    log(f"saved best checkpoint to {args.checkpoint}")


def parse_args():
    """Define the few command-line knobs needed for a small experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/cats_dogs", help="folder containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adam", "adamw"),
        default="adamw",
        help="optimizer used for parameter updates (default: adamw)",
    )
    parser.add_argument("--checkpoint", default="models/cnn_cats_dogs.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto selects CUDA when available; cpu forces CPU; cuda requires CUDA (default: auto)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Keeping this guard means importing train.py does not start training.  A
    # command-line invocation parses arguments and then enters the loop above.
    train(parse_args())

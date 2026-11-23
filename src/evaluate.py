"""Evaluate a saved CNN checkpoint on every image in the validation split.

Run from the repository root with ``python -m src.evaluate --help``. The
script reports overall and per-class metrics, a confusion matrix, and every
misclassified image. It also saves the complete report as JSON in ``models/``,
with ``_eval`` appended to the checkpoint's base name.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:  # Works with both ``python -m src.evaluate`` and ``python src/evaluate.py``.
    from .cnn import SimpleCNN
    from .dataset import ImageFolderDataset
    from .logging_utils import log
except ImportError:  # pragma: no cover - direct-file execution fallback.
    from cnn import SimpleCNN
    from dataset import ImageFolderDataset
    from logging_utils import log


def evaluate_checkpoint(checkpoint_path: Path, data_root: Path, batch_size: int, device: torch.device) -> dict:
    """Evaluate the checkpoint on the complete ``data_root/val`` split."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    image_size = checkpoint["image_size"]
    dataset = ImageFolderDataset(data_root / "val", image_size)
    if dataset.classes != classes:
        raise ValueError(
            "Validation class folders do not match checkpoint classes: "
            f"dataset={dataset.classes}, checkpoint={classes}"
        )
    if not dataset.samples:
        raise ValueError(f"No validation images found in {data_root / 'val'}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = SimpleCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    total_loss = 0.0
    correct = 0
    evaluated = 0
    misclassified = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            total_loss += criterion(logits, labels).item()
            correct += (predictions == labels).sum().item()
            evaluated += labels.size(0)
            pairs = labels.cpu() * len(classes) + predictions.cpu()
            confusion += torch.bincount(
                pairs, minlength=len(classes) * len(classes)
            ).reshape(len(classes), len(classes))

            for row, (truth, prediction) in enumerate(zip(labels.cpu().tolist(), predictions.cpu().tolist())):
                if truth == prediction:
                    continue
                sample_path = dataset.samples[evaluated - labels.size(0) + row][0]
                misclassified.append(
                    {
                        "image": sample_path.relative_to(data_root).as_posix(),
                        "actual": classes[truth],
                        "predicted": classes[prediction],
                        "predicted_confidence": float(probabilities[row, prediction].item()),
                        "probabilities": {
                            class_name: float(probabilities[row, index].item())
                            for index, class_name in enumerate(classes)
                        },
                    }
                )

    true_positives = confusion.diag().to(torch.float64)
    predicted_counts = confusion.sum(dim=0).to(torch.float64)
    actual_counts = confusion.sum(dim=1).to(torch.float64)
    precision = true_positives / predicted_counts.clamp_min(1)
    recall = true_positives / actual_counts.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    per_class = {}
    for index, class_name in enumerate(classes):
        per_class[class_name] = {
            "support": int(actual_counts[index].item()),
            "correct": int(true_positives[index].item()),
            "precision": float(precision[index].item()),
            "recall": float(recall[index].item()),
            "f1": float(f1[index].item()),
        }

    return {
        "checkpoint": str(checkpoint_path),
        "dataset": str(data_root),
        "split": "val",
        "image_size": image_size,
        "training": {
            "optimizer": checkpoint.get("optimizer"),
            "learning_rate": checkpoint.get("learning_rate"),
        },
        "num_samples": evaluated,
        "overall": {
            "loss": total_loss / evaluated,
            "accuracy": correct / evaluated,
            "correct": correct,
            "incorrect": evaluated - correct,
            "macro_precision": float(precision.mean().item()),
            "macro_recall": float(recall.mean().item()),
            "macro_f1": float(f1.mean().item()),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "rows": "actual classes",
            "columns": "predicted classes",
            "class_order": classes,
            "values": confusion.tolist(),
        },
        "misclassified_images": misclassified,
    }


def log_report(report: dict) -> None:
    """Print the report's summary, class metrics, confusion matrix, and errors."""
    overall = report["overall"]
    training = report["training"]
    log(
        f"training_optimizer={training['optimizer']} "
        f"learning_rate={training['learning_rate']}"
    )
    log(
        f"samples={report['num_samples']} image_size={report['image_size']} "
        f"loss={overall['loss']:.4f} accuracy={overall['accuracy']:.1%} "
        f"correct={overall['correct']} incorrect={overall['incorrect']}"
    )
    log(
        f"macro_precision={overall['macro_precision']:.1%} "
        f"macro_recall={overall['macro_recall']:.1%} macro_f1={overall['macro_f1']:.1%}"
    )
    for class_name, metrics in report["per_class"].items():
        log(
            f"class={class_name} support={metrics['support']} correct={metrics['correct']} "
            f"precision={metrics['precision']:.1%} recall={metrics['recall']:.1%} "
            f"f1={metrics['f1']:.1%}"
        )
    matrix = report["confusion_matrix"]
    log(
        f"confusion_matrix rows=actual columns=predicted "
        f"classes={matrix['class_order']} values={matrix['values']}"
    )
    if report["misclassified_images"]:
        log(f"misclassified_images={len(report['misclassified_images'])}")
        for item in report["misclassified_images"]:
            log(
                f"misclassified image={item['image']} actual={item['actual']} "
                f"predicted={item['predicted']} "
                f"confidence={item['predicted_confidence']:.1%} "
                f"probabilities={item['probabilities']}"
            )
    else:
        log("misclassified_images=0")


def parse_args():
    """Define the checkpoint, validation dataset, batch size, and device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/cnn_cats_dogs.pt")
    parser.add_argument("--data", default="data/cats_dogs_resized_64", help="dataset root containing val/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto selects CUDA when available; cpu forces CPU; cuda requires CUDA",
    )
    return parser.parse_args()


def main(args) -> None:
    """Run evaluation, print results, and save a JSON report in ``models/``."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot access CUDA")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data).resolve()
    log(f"evaluation_device={device}")
    report = evaluate_checkpoint(checkpoint_path, data_root, args.batch_size, device)
    report["evaluation_device"] = str(device)
    log_report(report)

    report_path = Path("models") / f"{checkpoint_path.stem}_eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    log(f"saved evaluation report to {report_path}")


if __name__ == "__main__":
    main(parse_args())

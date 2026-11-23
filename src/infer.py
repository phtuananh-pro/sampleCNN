"""Classify one image with a checkpoint produced by :mod:`src.train`.

Inference is the chapter's forward pass without a target or an update:

    image -> same preprocessing -> trained CNN -> logits -> probabilities

The checkpoint stores the class names and image size used during training so
the inference path does not have to guess how to interpret the output vector.
The logits-to-probabilities step follows Chapter VI.3, Section 15.
"""

import argparse

import torch

try:  # Works both with ``python -m src.infer`` and ``python src/infer.py``.
    from .cnn import SimpleCNN
    from .dataset import load_image
    from .logging_utils import log, log_model_info
except ImportError:  # pragma: no cover - only used for direct file execution.
    from cnn import SimpleCNN
    from dataset import load_image
    from logging_utils import log, log_model_info


def parse_args():
    """Define the checkpoint, input image, and optional device override."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/cnn_cats_dogs.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main(args):
    """Load a trained model and print its prediction for one image."""
    # Choose the same kind of device used by training.  map_location below
    # ensures that a checkpoint trained on a GPU can still be loaded on a CPU.
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # A checkpoint is a small dictionary, not just raw weights.  It contains:
    # * model_state: learned convolution, bias, and classifier parameters;
    # * classes: integer-index-to-name mapping such as ["cat", "dog"];
    # * image_size: preprocessing size expected by the convolution layers.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]

    # Recreate the same architecture with the same number of output logits,
    # then copy in the learned parameters from training.
    model = SimpleCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # Evaluation mode prevents training-only behavior.  There is no optimizer
    # here because inference should not change the model.
    model.eval()

    if device.type == "cuda":
        log(f"loaded checkpoint; gpu={torch.cuda.get_device_name(device)}")
    else:
        log("loaded checkpoint")
    log_model_info(model, classes, checkpoint["image_size"], device, args.checkpoint)

    # ``load_image`` returns (1, 3, H, W): one image in a batch.  The model's
    # output has shape (1, num_classes); [0] below selects that one row.
    with torch.no_grad():
        logits = model(load_image(args.image, checkpoint["image_size"]).to(device))

        # Softmax turns arbitrary logits into nonnegative values summing to one.
        # These values are useful for display as class probabilities.  Softmax
        # was intentionally omitted from SimpleCNN.forward because training's
        # CrossEntropyLoss consumes logits directly.
        probabilities = torch.softmax(logits, dim=1)[0]

    # The largest probability is the model's predicted class.  Converting the
    # scalar tensor to int lets us use it as a normal Python list index.
    predicted = int(probabilities.argmax())
    log(f"prediction: {classes[predicted]} ({probabilities[predicted].item():.1%})")

    # Print every class so the learner can see the whole output distribution,
    # not only the winner selected by argmax.
    for class_name, probability in zip(classes, probabilities.tolist()):
        log(f"  {class_name}: {probability:.1%}")


if __name__ == "__main__":
    # Direct execution is useful for a quick experiment:
    # python src/infer.py --image path/to/image.jpg
    main(parse_args())

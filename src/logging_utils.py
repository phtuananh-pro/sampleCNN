"""Small console logging helpers shared by training and inference."""

from datetime import datetime


def log(message: str) -> None:
    """Print one console line prefixed with the local 24-hour time."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def log_model_info(model, classes, image_size: int, device, checkpoint: str | None = None) -> None:
    """Print a compact summary of the model, its inputs, and its runtime."""
    parameters = list(model.parameters())
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)

    log(f"model={model.__class__.__name__}")
    log(f"parameters={total_parameters:,} trainable={trainable_parameters:,}")
    log(f"classes={classes}")
    log(f"input_shape=(3, {image_size}, {image_size}) RGB")
    log(f"device={device}")
    if checkpoint is not None:
        log(f"checkpoint={checkpoint}")

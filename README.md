# Simplest workable CNN

This repository is a small learning example for the CNN ideas in [Frontier Models guide](https://frontiermodels.guide) `chapter VI.3 on CNN`:

```text
image -> shared local kernels -> ReLU -> pooling -> feature hierarchy -> classifier
```

The example classifies cats and dogs. It is intentionally simple: it uses a few convolution layers, no pretrained model, no augmentation library, and no performance-focused tricks. The goal is to make the chapter’s ideas visible in runnable code, not to achieve state-of-the-art accuracy.

The model demonstrates:

- RGB images as `C x H x W` tensors.
- Convolution kernels that inspect local regions and reuse the same weights at every position.
- Feature channels, ReLU, max-pooling, increasing channel width, and decreasing spatial resolution.
- A backbone followed by global average pooling and a classification head.
- The complete loop: forward pass, cross-entropy loss, backpropagation, and a selectable SGD, Adam, or AdamW update.

## Install

Use Python 3.11 or newer. The Windows and Linux requirements install the CUDA 13.2 PyTorch wheel; the macOS requirement uses the regular PyTorch package.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements file selects the CUDA-enabled PyTorch build on Windows and Linux, so training can use an available NVIDIA GPU by default. Verify the installation in the same environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

On Windows or Linux with a supported NVIDIA GPU and driver, `torch.cuda.is_available()` should print `True`. PyTorch 2.15 provides CUDA 13.2 wheels for Windows and Linux. The CUDA runtime is supplied by the PyTorch wheel; install an up-to-date NVIDIA driver on the machine.

Only PyTorch, Pillow, and NumPy are required. The image loader does not use `torchvision`.

## Dataset layout

The small dataset folder structure in this repository is:

```text
data/cats_dogs/
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

The loader accepts any image dataset with one folder per class. Images are resized to 64x64 and converted to RGB tensors.

### Create a small dataset

- [Microsoft Cats and Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765) is a straightforward source. Keep only a small number of readable images from its `Cat/` and `Dog/` folders.
- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) contains cat and dog breeds. Use a small subset for a short experiment.

If you download raw folders such as `data/raw/PetImages/Cat/` and `data/raw/PetImages/Dog/`, create the expected layout with:

```bash
python -m src.prepare_data \
  --source data/raw/PetImages \
  --output data/cats_dogs \
  --per-class 100 \
  --val-count 20
```

The helper validates images, chooses a deterministic subset, and copies the first images into `train/` and `val/` folders. It does not delete existing files; choose a new output folder or clear an old generated dataset yourself before regenerating it.

## Train

### Step 1: Resize the source images

Run this from the repository root, using the dataset folder that contains `train/` and `val/`:

```bash
python -m src.prepare_images --data data/cats_dogs --image-size 64
```

By default, this creates `data/cats_dogs_resized_64`, preserving the `train/` and `val/` class-folder structure. The script resizes images to 64x64 and records source file size and modification time in a manifest. On later runs, it reuses unchanged cached images and resizes only new or changed images. Use `--output` to select a different cache folder.

For example, to create a 128x128 cache instead:

```bash
python -m src.prepare_images --data data/cats_dogs --image-size 128
```

### Step 2: Train using the resized images

Point `--data` at the resized cache folder and set `--image-size` to the same size used when resizing. For the 64x64 cache:

```bash
python -m src.train --data data/cats_dogs_resized_64 --epochs 20 --image-size 64
```

For the 128x128 cache, use:

```bash
python -m src.train --data data/cats_dogs_resized_128 --epochs 20 --image-size 128
```

When training on the resized cache, the loader reads those image dimensions directly and does not resize the images again. You can also train from the original `data/cats_dogs` folder; in that case, the loader resizes images as it reads them.

The best validation checkpoint is written to `models/cnn_cats_dogs.pt`. Useful options:

```bash
python -m src.train --help
python -m src.train --data data/cats_dogs --epochs 5 --batch-size 16 --device cpu
python -m src.train --data data/cats_dogs --epochs 5 --device cuda
python -m src.train --data data/cats_dogs_resized_64 --optimizer sgd
python -m src.train --data data/cats_dogs_resized_64 --optimizer adam
python -m src.train --data data/cats_dogs_resized_64 --optimizer adamw
```

`--device` accepts `auto` (the default), `cpu`, or `cuda`. `auto` uses CUDA if the installed PyTorch build can access it, otherwise it falls back to CPU. Explicitly requesting `cuda` stops with an error if CUDA is unavailable.

An NVIDIA GPU and driver do not by themselves enable CUDA in PyTorch. If `torch.cuda.is_available()` is `False` after installing `requirements.txt`, check that you used Python 3.11 or newer and that the installed PyTorch version ends in `+cu132`. The [official PyTorch installation page](https://pytorch.org/get-started/locally/) lists the currently supported CUDA wheels.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Each epoch reports validation accuracy, macro F1, per-class precision and recall, and a confusion matrix whose rows are actual classes and columns are predicted classes. The training command prints the selected device and, when using CUDA, the GPU name. The default model uses 64x64 images, three 3x3 convolution layers, max-pooling after each layer, and global average pooling. CPU training is expected to be usable but modest; fewer epochs and fewer images make a quick smoke test.

### Step 3: Evaluate a saved model

After training, evaluate the checkpoint against every image in the resized validation split:

```bash
python -m src.evaluate --checkpoint models/cnn_cats_dogs.pt --data data/cats_dogs_resized_64
```

The evaluator uses the image size stored in the checkpoint and reports loss, accuracy, macro precision/recall/F1, per-class precision/recall/F1 and support, the confusion matrix, and each misclassified image with its predicted probabilities. It saves the same results as JSON to `models/cnn_cats_dogs_eval.json`. If you use a different checkpoint name, the report uses that name with `_eval.json` appended (for example, `models/experiment_eval.json`). Use `--device cpu` to force CPU evaluation or `--device cuda` to require CUDA.

## Inference

Classify one image with the saved checkpoint:

```bash
python -m src.infer --checkpoint models/cnn_cats_dogs.pt --image data/cats_dogs/val/cat/0000.jpg
```

The output includes the predicted class, its confidence, and all class probabilities.

## Code map

- [`src/cnn.py`](src/cnn.py): the small CNN backbone and classifier.
- [`src/dataset.py`](src/dataset.py): minimal Pillow/PyTorch image-folder loader.
- [`src/train.py`](src/train.py): training loop, validation, and checkpointing.
- [`src/evaluate.py`](src/evaluate.py): standalone evaluation of a checkpoint on the full validation split.
- [`src/infer.py`](src/infer.py): single-image inference.
- [`src/logging_utils.py`](src/logging_utils.py): timestamped console output and model summary.
- [`src/prepare_data.py`](src/prepare_data.py): make a small validated subset from raw class folders.
- [`src/prepare_images.py`](src/prepare_images.py): resize images once and reuse an incremental local cache.

Read the guide alongside `src/cnn.py` and trace the tensor shape after each pooling layer. That is the shortest route from the chapter’s diagrams and equations to a working CNN.

## Training arguments

Run `python -m src.train --help` to display the command-line help. Paths are resolved from the current working directory; run the command from the repository root.

Prepare or update the resized-image cache with `python -m src.prepare_images --help`:

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/cats_dogs` | Source dataset with `train/` and `val/` splits. |
| `--output` | `data/cats_dogs_resized_64` | Cache folder; by default its name is derived from the source folder and image size. |
| `--image-size` | `64` | Target square width and height. |
| `-h`, `--help` | - | Show help and exit. |

For a different image size, such as 128, use a matching cache folder and pass `--image-size 128` when training.

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/cats_dogs` | Raw or resized dataset root containing `train/` and `val/`, each with one folder per class. |
| `--epochs` | `20` | Number of complete passes through the training set. |
| `--batch-size` | `32` | Number of images processed before each optimizer update. |
| `--image-size` | `64` | Resize images to this square width and height. |
| `--learning-rate` | `0.001` | Learning rate passed to the selected optimizer. |
| `--optimizer` | `adamw` | Optimizer: `sgd`, `adam`, or `adamw`. |
| `--checkpoint` | `models/cnn_cats_dogs.pt` | File path for the best validation checkpoint. |
| `--seed` | `7` | Seed for Python and PyTorch random-number generators. |
| `--device` | `auto` | Choose `auto`, `cpu`, or `cuda`. `auto` uses CUDA when available and otherwise uses CPU; `cuda` errors if unavailable. |
| `-h`, `--help` | — | Show help and exit. |

Choose `--optimizer sgd`, `--optimizer adam`, or `--optimizer adamw`. AdamW remains the default, with a learning rate of `0.001`; the chapter derives the simpler SGD update, while Adam and AdamW are also available for comparison.

Example:

```bash
python -m src.train --data data/cats_dogs_resized_64 --epochs 20 --batch-size 32 --device auto
```

## Evaluation arguments

Run `python -m src.evaluate --help` to display the command-line help. The evaluator processes the complete `val/` split under `--data` and saves its JSON report beside the checkpoint.

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `models/cnn_cats_dogs.pt` | Trained model checkpoint to evaluate. |
| `--data` | `data/cats_dogs_resized_64` | Resized dataset root containing `val/`. |
| `--batch-size` | `32` | Number of validation images processed per batch. |
| `--device` | `auto` | Choose `auto`, `cpu`, or `cuda`. |
| `-h`, `--help` | - | Show help and exit. |

## Inference arguments

Run `python -m src.infer --help` to display the command-line help. The checkpoint supplies the model's class names and image size, so these do not need separate arguments.

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | `models/cnn_cats_dogs.pt` | Checkpoint created by `src.train`. |
| `--image` | Required | Image file to classify. |
| `--cpu` | Off | Force CPU inference, even when CUDA is available. Without this option, CUDA is selected automatically when available. |
| `-h`, `--help` | — | Show help and exit. |

Example:

```bash
python -m src.infer --checkpoint models/cnn_cats_dogs.pt --image data/cats_dogs/val/cat/0000.jpg
```

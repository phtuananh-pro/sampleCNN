# Simplest workable CNN

This repository is a small learning example for the CNN ideas in [Frontier Models guide](https://frontiermodels.guide) `chapter VI.2 on CNN`:

```text
image -> shared local kernels -> ReLU -> pooling -> feature hierarchy -> classifier
```

The example classifies cats and dogs. It is intentionally simple: it uses a few convolution layers, no pretrained model, no augmentation library, and no performance-focused tricks. The goal is to make the chapter’s ideas visible in runnable code, not to achieve state-of-the-art accuracy.

The model demonstrates:

- RGB images as `C x H x W` tensors.
- Convolution kernels that inspect local regions and reuse the same weights at every position.
- Feature channels, ReLU, max-pooling, increasing channel width, and decreasing spatial resolution.
- A backbone followed by global average pooling and a classification head.
- The complete loop: forward pass, cross-entropy loss, backpropagation, and an SGD update.

## Install

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Only PyTorch and Pillow are required.

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

From the repository root:

```bash
python -m src.train --data data/cats_dogs --epochs 10
```

```pwsh
python .\src\train.py --data .\data\cats_dogs --epochs 10
```

The best validation checkpoint is written to `models/cnn_cats_dogs.pt`. Useful options:

```bash
python -m src.train --help
python -m src.train --data data/cats_dogs --epochs 5 --batch-size 16 --cpu
```

The default model uses 64x64 images, three 3x3 convolution layers, max-pooling after each layer, and global average pooling. CPU training is expected to be usable but modest; fewer epochs and fewer images make a quick smoke test.

## Inference

Classify one image with the saved checkpoint:

```bash
python -m src.infer \
  --checkpoint models/cnn_cats_dogs.pt \
  --image data/cats_dogs/val/cat/0000.jpg
```
```

The output includes the predicted class, its confidence, and all class probabilities.

## Code map

- [`src/cnn.py`](src/cnn.py): the small CNN backbone and classifier.
- [`src/dataset.py`](src/dataset.py): minimal Pillow/PyTorch image-folder loader.
- [`src/train.py`](src/train.py): training loop, validation, and checkpointing.
- [`src/infer.py`](src/infer.py): single-image inference.
- [`src/prepare_data.py`](src/prepare_data.py): make a small validated subset from raw class folders.

Read the guide alongside `src/cnn.py` and trace the tensor shape after each pooling layer. That is the shortest route from the chapter’s diagrams and equations to a working CNN.

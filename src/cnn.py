"""The small CNN used by this repository.

This file is intentionally written as a teaching companion to the CNN chapter.
The important ideas are visible in the order of the modules:

    image -> convolution -> ReLU -> pooling -> ... -> global average -> logits

The code uses PyTorch's ``Conv2d`` rather than implementing convolution with
explicit Python loops.  ``Conv2d`` still performs the same conceptual operation
described in the chapter: each learned kernel looks at a small local window,
and the same kernel weights are reused at every spatial position.

Useful cross-references while reading the chapter:

* Sections 1-7: locality, kernels, feature maps, and feature channels.
* Sections 8-11: stages, downsampling, backbone/head, and classification.
* Sections 15-17: forward computation, loss, backpropagation, and updates.
* Section 22: receptive fields growing through repeated local operations.
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """A deliberately small convolutional image classifier.

    The model has two logical parts, matching the chapter's backbone/head split:

    * ``backbone`` turns pixels into a hierarchy of learned feature maps.
    * ``pool`` and ``classifier`` turn the final feature maps into class logits.

    For the default 64x64 RGB input, the tensor shapes are:

        input       : (B,  3, 64, 64)  measured RGB channels
        after block 1: (B, 16, 32, 32)  local features, half resolution
        after block 2: (B, 32, 16, 16)  more channels, half resolution again
        after block 3: (B, 64,  8,  8)  deeper features, half resolution again
        after pool   : (B, 64,  1,  1)  one average per feature channel
        flattened    : (B, 64)         one image-level feature vector
        logits       : (B, num_classes)

    ``B`` is the batch size.  The spatial dimensions shrink while the number
    of feature channels grows.  This is a simple example of the chapter's
    backbone pattern: preserve useful information in fewer spatial positions,
    while giving each remaining position a richer feature vector.
    """

    def __init__(self, num_classes: int):
        # ``nn.Module`` stores parameters and child layers for us.  Calling the
        # parent constructor is what lets PyTorch find the convolution and
        # linear-layer weights when we create an optimizer later.
        super().__init__()

        # ------------------------------------------------------------------
        # The backbone: three simple convolutional stages.
        # ------------------------------------------------------------------
        #
        # Each Conv2d has weights shaped conceptually like:
        #
        #     (output_channels, input_channels, kernel_height, kernel_width)
        #
        # For example, the first layer has 16 learned 3x3 kernels.  Because
        # the input is RGB, each output kernel has a small 3x3 slice for each
        # of the 3 input channels: 3 * 3 * 3 weights, plus one bias.  The
        # kernel is scanned over the image and reused at every location.  That
        # is the chapter's combination of locality and weight sharing.
        #
        # ``padding=1`` surrounds the input with one zero-valued border.  With
        # a 3x3 kernel and stride 1, this keeps height and width unchanged:
        # 64x64 stays 64x64 before pooling.
        #
        # ReLU applies max(0, value) element by element.  It is not a learned
        # layer; it adds the nonlinearity needed for several convolution layers
        # to compose into useful patterns instead of one large linear mapping.
        #
        # MaxPool2d(2) is a fixed local summary: each non-overlapping 2x2
        # window keeps its largest activation.  Its stride is 2, so it also
        # downsamples height and width by about two.  This gives later kernels
        # a larger receptive field in the original image coordinates.
        #
        # The channel progression 3 -> 16 -> 32 -> 64 has a clear meaning:
        # the input's three channels are measured RGB values, while the hidden
        # channels are learned feature coordinates.  They are not required to
        # correspond to human names such as "edge" or "ear".
        self.backbone = nn.Sequential(
            # (B,  3, 64, 64) -> (B, 16, 64, 64)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            # A feature map can contain positive or negative responses before
            # ReLU.  ReLU keeps positive evidence and clips negative evidence.
            # Shape is unchanged: (B, 16, 64, 64).
            nn.ReLU(),
            # (B, 16, 64, 64) -> (B, 16, 32, 32)
            nn.MaxPool2d(kernel_size=2),

            # The second layer reads all 16 incoming feature channels in every
            # local 3x3 window and produces 32 new feature channels.  It can
            # combine simple first-layer patterns into slightly richer ones.
            # (B, 16, 32, 32) -> (B, 32, 32, 32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # (B, 32, 32, 32) -> (B, 32, 32, 32)
            nn.ReLU(),
            # (B, 32, 32, 32) -> (B, 32, 16, 16)
            nn.MaxPool2d(kernel_size=2),

            # The third layer repeats the same local-and-shared operation at a
            # deeper level.  Its input is no longer raw RGB; it is a grid of
            # learned feature vectors produced by the earlier stages.
            # (B, 32, 16, 16) -> (B, 64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # (B, 64, 16, 16) -> (B, 64, 16, 16)
            nn.ReLU(),
            # (B, 64, 16, 16) -> (B, 64, 8, 8)
            nn.MaxPool2d(kernel_size=2),
        )

        # Classification needs one image-level representation rather than a
        # separate prediction at every remaining spatial position.  Adaptive
        # average pooling averages each of the 64 feature maps over its whole
        # 8x8 grid.  The result is always 1x1 per channel, which keeps the
        # classifier independent of small input-size changes.
        #
        # This is the chapter's global average pooling step.  It intentionally
        # compresses location for an image-level task such as cat vs dog.  A
        # segmentation or OCR head would usually preserve more spatial detail.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # The final linear layer is the classification head.  It maps the 64
        # pooled feature values to one unnormalized score (a logit) per class.
        # CrossEntropyLoss in train.py applies the appropriate normalization
        # internally, so this model should return logits rather than softmax
        # probabilities during training.
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        """Run one batch through the CNN and return class logits.

        Parameters
        ----------
        x:
            A float tensor with shape ``(B, 3, H, W)`` and values scaled to
            approximately ``[0, 1]``.  The dataset creates exactly this
            channel-first representation from each RGB image.

        Returns
        -------
        torch.Tensor
            A tensor with shape ``(B, num_classes)``.  Each row contains the
            model's scores for one image.  The largest score is the predicted
            class; probabilities are only needed when displaying a prediction.
        """
        # The backbone keeps a spatial grid while repeatedly applying local
        # shared kernels, nonlinearities, and downsampling.
        x = self.backbone(x)

        # Global average pooling changes (B, 64, 8, 8) into (B, 64, 1, 1).
        x = self.pool(x)

        # ``flatten(1)`` keeps the batch dimension and flattens everything
        # after it: (B, 64, 1, 1) -> (B, 64).
        x = x.flatten(1)

        # The task head converts the image-level feature vector into logits.
        return self.classifier(x)

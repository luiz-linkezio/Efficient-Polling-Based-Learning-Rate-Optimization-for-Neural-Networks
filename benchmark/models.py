"""The two networks every method trains: the plainest ones that fit the data.

Neither has batch norm or dropout, so the optimizer is the only thing in a run
that adapts, and a difference between two methods is a difference in how they
choose the learning rate.
"""

from __future__ import annotations

import torch
from torch import nn

from .datasets import DatasetSpec

__all__ = ["SimpleCNN", "SimpleMLP", "build_model"]


class SimpleCNN(nn.Module):
    """5-layer CNN, 557,898 parameters on CIFAR-10, the width the paper quotes.

    The global pool before the classifier means the input side length never
    enters: 28x28 and 32x32 both arrive at the same 256 features, so only the
    channel count and the class count vary between datasets.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleMLP(nn.Module):
    """The dense counterpart of :class:`SimpleCNN`, for data with no spatial
    axes to convolve over, such as Covertype's 54 features."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        widths: tuple[int, ...] = (512, 256, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_features
        for width in widths:
            layers += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(spec: DatasetSpec) -> nn.Module:
    """The network the dataset's shape calls for: the CNN for images, the MLP otherwise."""
    if spec.is_image:
        return SimpleCNN(in_channels=spec.in_channels, num_classes=spec.num_classes)
    (features,) = spec.input_shape
    return SimpleMLP(in_features=features, num_classes=spec.num_classes)

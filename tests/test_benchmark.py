"""What the sweep script builds for a given dataset.

The model and the divergence threshold both depend on the dataset, and both
fail quietly if they don't: a threshold left at CIFAR-10's ten classes sits
*below* the loss a hundred-class run starts at, so every batch would look like
a blow-up and the run would restart forever without erroring once.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from benchmark import (
    build_epoch_polling,
    build_model,
    build_optimizer,
    checkpoint_path,
    parse_args,
    result_path,
)
from benchmark_datasets import spec_for


def args_for(dataset: str, *extra: str):
    return parse_args(["--data-dir", "/nowhere", "--dataset", dataset, *extra])


@pytest.mark.parametrize(
    ("dataset", "channels", "side", "classes"),
    [
        ("cifar10", 3, 32, 10),
        ("cifar100", 3, 32, 100),
        ("mnist", 1, 28, 10),
        ("fashion_mnist", 1, 28, 10),
    ],
)
def test_the_model_takes_the_shape_the_dataset_has(
    dataset: str, channels: int, side: int, classes: int
) -> None:
    model = build_model(spec_for(dataset))

    logits = model(torch.zeros(2, channels, side, side))

    assert logits.shape == (2, classes)


def test_the_cifar10_model_is_still_the_one_the_paper_counts() -> None:
    """557,898 parameters is quoted in the paper; generalizing the model must
    not have moved it."""
    model = build_model(spec_for("cifar10"))

    assert sum(p.numel() for p in model.parameters()) == 557_898


def test_the_divergence_threshold_follows_the_class_count() -> None:
    spec = spec_for("cifar100")
    model = build_model(spec)
    optimizer, _ = build_optimizer("efficient", model, args_for("cifar100"), 42, spec)

    assert optimizer.rollback_loss == pytest.approx(2.0 * math.log(100))


def test_the_relative_variant_gets_the_same_threshold() -> None:
    spec = spec_for("cifar100")
    model = build_model(spec)
    optimizer, _ = build_optimizer("efficient_relative", model, args_for("cifar100"), 42, spec)

    assert optimizer.rollback_loss == pytest.approx(2.0 * math.log(100))


def test_the_epoch_controller_gets_the_same_threshold() -> None:
    spec = spec_for("cifar100")
    controller = build_epoch_polling("efficient_relative_epoch", args_for("cifar100"), spec)

    assert controller.rollback_loss == pytest.approx(2.0 * math.log(100))


def test_cifar10_keeps_the_threshold_the_recorded_runs_used() -> None:
    spec = spec_for("cifar10")
    model = build_model(spec)
    optimizer, _ = build_optimizer("efficient", model, args_for("cifar10"), 42, spec)

    assert optimizer.rollback_loss == pytest.approx(2.0 * math.log(10))


def test_results_land_in_a_directory_of_their_own_per_dataset() -> None:
    assert Path(args_for("mnist").results_dir) == Path("results/mnist")
    assert Path(args_for("cifar10").results_dir) == Path("results/cifar10")


def test_an_explicit_results_dir_still_wins() -> None:
    args = args_for("mnist", "--results-dir", "results/mnist-lr1e-5")

    assert Path(args.results_dir) == Path("results/mnist-lr1e-5")
    assert result_path(args, "efficient", 42).name == "efficient_seed42.json"


def test_checkpoints_of_two_datasets_do_not_overwrite_each_other() -> None:
    mnist = checkpoint_path(args_for("mnist"), spec_for("mnist"), "efficient", 42)
    cifar = checkpoint_path(args_for("cifar10"), spec_for("cifar10"), "efficient", 42)

    assert mnist != cifar
    assert "mnist" in mnist.name


def test_a_tabular_dataset_gets_a_dense_network_not_a_cnn() -> None:
    """Covertype has no spatial axes at all: the convolutional stack cannot be
    what runs there, and the claim is about the learning rate, not the model."""
    spec = spec_for("covertype")

    logits = build_model(spec)(torch.zeros(2, 54))

    assert logits.shape == (2, 7)


def test_the_tabular_threshold_follows_its_seven_classes() -> None:
    spec = spec_for("covertype")
    optimizer, _ = build_optimizer("efficient", build_model(spec), args_for("covertype"), 42, spec)

    assert optimizer.rollback_loss == pytest.approx(2.0 * math.log(7))

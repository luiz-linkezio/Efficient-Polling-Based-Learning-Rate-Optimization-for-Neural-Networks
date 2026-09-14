"""The loaders are the one place where a silent parsing bug would corrupt every
number in a sweep, so they are checked against files built here byte by byte --
no download, no network, nothing on disk that the test did not write."""

from __future__ import annotations

import gzip
import pickle
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark.datasets import (
    DATASETS,
    ArrayDataset,
    build_split,
    compute_mean_std,
    load_split,
    spec_for,
)


def _write_cifar_batch(
    path: Path, images: np.ndarray, labels: list[int], label_key: bytes = b"labels"
) -> None:
    """One pickled CIFAR batch: (N, 3072) uint8 rows plus a list of labels."""
    path.write_bytes(pickle.dumps({b"data": images, label_key: labels}))


def _idx_bytes(array: np.ndarray) -> bytes:
    """The IDX container: magic, one big-endian int32 per dimension, then bytes."""
    magic = struct.pack(">I", 0x00000800 + array.ndim)
    dims = b"".join(struct.pack(">I", d) for d in array.shape)
    return magic + dims + array.astype(np.uint8).tobytes()


@pytest.fixture
def cifar10_dir(tmp_path: Path) -> Path:
    for i in range(1, 6):
        images = np.full((2, 3072), i, dtype=np.uint8)
        _write_cifar_batch(tmp_path / f"data_batch_{i}", images, [i % 10, (i + 1) % 10])
    _write_cifar_batch(tmp_path / "test_batch", np.zeros((3, 3072), dtype=np.uint8), [0, 1, 2])
    return tmp_path


@pytest.fixture
def mnist_dir(tmp_path: Path) -> Path:
    for split, n in (("train", 4), ("t10k", 2)):
        images = np.arange(n * 28 * 28, dtype=np.uint8).reshape(n, 28, 28)
        (tmp_path / f"{split}-images-idx3-ubyte").write_bytes(_idx_bytes(images))
        labels = np.arange(n, dtype=np.uint8) % 10
        (tmp_path / f"{split}-labels-idx1-ubyte").write_bytes(_idx_bytes(labels))
    return tmp_path


# --- the registry ------------------------------------------------------------


def test_every_registered_dataset_declares_its_shape() -> None:
    for key, spec in DATASETS.items():
        assert spec.key == key
        assert spec.num_classes >= 2
        assert spec.input_shape
        assert all(d > 0 for d in spec.input_shape)
        assert spec.is_image == (len(spec.input_shape) == 3)
        assert spec.source.startswith("http")


def test_an_image_spec_reports_channels_and_side_length() -> None:
    spec = spec_for("cifar10")

    assert spec.input_shape == (3, 32, 32)
    assert spec.is_image
    assert (spec.in_channels, spec.image_size) == (3, 32)


def test_a_tabular_spec_reports_a_feature_count_and_no_side_length() -> None:
    spec = spec_for("covertype")

    assert spec.input_shape == (54,)
    assert not spec.is_image
    with pytest.raises(ValueError, match="not an image"):
        _ = spec.image_size


def test_an_unknown_dataset_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="cifar10"):
        spec_for("cifar11")


# --- CIFAR -------------------------------------------------------------------


def test_cifar10_train_concatenates_the_five_batches(cifar10_dir: Path) -> None:
    images, labels = load_split("cifar10", cifar10_dir, train=True)

    assert images.shape == (10, 3, 32, 32)
    assert images.dtype == np.uint8
    assert labels.tolist() == [1, 2, 2, 3, 3, 4, 4, 5, 5, 6]


def test_cifar10_test_split_reads_the_single_test_batch(cifar10_dir: Path) -> None:
    images, labels = load_split("cifar10", cifar10_dir, train=False)

    assert images.shape == (3, 3, 32, 32)
    assert labels.tolist() == [0, 1, 2]


def test_cifar100_reads_the_fine_labels(tmp_path: Path) -> None:
    """The coarse labels are a 20-class relabelling of the same images; taking
    them by accident would quietly turn this into a different benchmark."""
    images = np.zeros((2, 3072), dtype=np.uint8)
    payload = {b"data": images, b"fine_labels": [7, 99], b"coarse_labels": [1, 2]}
    (tmp_path / "train").write_bytes(pickle.dumps(payload))

    _, labels = load_split("cifar100", tmp_path, train=True)

    assert labels.tolist() == [7, 99]
    assert spec_for("cifar100").num_classes == 100


def test_a_missing_cifar_batch_names_the_directory_to_point_at(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cifar-10-batches-py"):
        load_split("cifar10", tmp_path, train=True)


# --- IDX ---------------------------------------------------------------------


def test_mnist_reads_the_idx_files_as_single_channel_images(mnist_dir: Path) -> None:
    images, labels = load_split("mnist", mnist_dir, train=True)

    assert images.shape == (4, 1, 28, 28)
    assert images.dtype == np.uint8
    assert labels.tolist() == [0, 1, 2, 3]
    assert images[0, 0, 0, 0] == 0
    assert images[1, 0, 0, 0] == (28 * 28) % 256


def test_fashion_mnist_uses_the_same_filenames_as_mnist(mnist_dir: Path) -> None:
    images, _ = load_split("fashion_mnist", mnist_dir, train=True)

    assert images.shape == (4, 1, 28, 28)


def test_gzipped_idx_files_are_read_as_downloaded(tmp_path: Path) -> None:
    """Both mirrors ship .gz; unpacking by hand is a step people skip."""
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    labels = np.array([3, 4], dtype=np.uint8)
    with gzip.open(tmp_path / "train-images-idx3-ubyte.gz", "wb") as f:
        f.write(_idx_bytes(images))
    with gzip.open(tmp_path / "train-labels-idx1-ubyte.gz", "wb") as f:
        f.write(_idx_bytes(labels))

    _, read = load_split("mnist", tmp_path, train=True)

    assert read.tolist() == [3, 4]


def test_the_dotted_idx_filename_variant_is_accepted(tmp_path: Path) -> None:
    """Some archives name it ``train-images.idx3-ubyte``."""
    images = np.zeros((1, 28, 28), dtype=np.uint8)
    (tmp_path / "train-images.idx3-ubyte").write_bytes(_idx_bytes(images))
    (tmp_path / "train-labels.idx1-ubyte").write_bytes(_idx_bytes(np.array([5], dtype=np.uint8)))

    _, labels = load_split("mnist", tmp_path, train=True)

    assert labels.tolist() == [5]


def test_a_truncated_idx_file_is_rejected(tmp_path: Path) -> None:
    full = _idx_bytes(np.zeros((4, 28, 28), dtype=np.uint8))
    (tmp_path / "train-images-idx3-ubyte").write_bytes(full[: len(full) - 10])
    (tmp_path / "train-labels-idx1-ubyte").write_bytes(_idx_bytes(np.zeros(4, dtype=np.uint8)))

    with pytest.raises(ValueError, match="truncated"):
        load_split("mnist", tmp_path, train=True)


def test_a_file_that_is_not_idx_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "train-images-idx3-ubyte").write_bytes(b"not an idx file at all")
    (tmp_path / "train-labels-idx1-ubyte").write_bytes(_idx_bytes(np.zeros(1, dtype=np.uint8)))

    with pytest.raises(ValueError, match="magic"):
        load_split("mnist", tmp_path, train=True)


def test_images_and_labels_of_different_lengths_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "train-images-idx3-ubyte").write_bytes(
        _idx_bytes(np.zeros((4, 28, 28), dtype=np.uint8))
    )
    (tmp_path / "train-labels-idx1-ubyte").write_bytes(_idx_bytes(np.zeros(3, dtype=np.uint8)))

    with pytest.raises(ValueError, match="3 labels"):
        load_split("mnist", tmp_path, train=True)


def test_images_of_the_wrong_side_length_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "train-images-idx3-ubyte").write_bytes(
        _idx_bytes(np.zeros((2, 32, 32), dtype=np.uint8))
    )
    (tmp_path / "train-labels-idx1-ubyte").write_bytes(_idx_bytes(np.zeros(2, dtype=np.uint8)))

    with pytest.raises(ValueError, match="28"):
        load_split("mnist", tmp_path, train=True)


# --- the tensor view ---------------------------------------------------------


def test_the_dataset_yields_float_images_and_long_labels(cifar10_dir: Path) -> None:
    ds = build_split("cifar10", cifar10_dir, train=False)
    image, label = ds[0]

    assert image.dtype == torch.float32
    assert image.shape == (3, 32, 32)
    assert label.dtype == torch.int64


def test_normalization_applies_the_statistics_it_was_given(mnist_dir: Path) -> None:
    mean = torch.full((1, 1, 1), 10.0)
    std = torch.full((1, 1, 1), 2.0)
    plain = build_split("mnist", mnist_dir, train=True)
    normalized = build_split("mnist", mnist_dir, train=True, mean=mean, std=std)

    assert torch.allclose(normalized[0][0], (plain[0][0] - 10.0) / 2.0)


def test_mean_and_std_must_be_passed_together(mnist_dir: Path) -> None:
    with pytest.raises(ValueError, match="together"):
        build_split("mnist", mnist_dir, train=True, mean=torch.zeros(1, 1, 1))


def test_statistics_are_computed_per_channel_and_ready_to_broadcast(
    cifar10_dir: Path,
) -> None:
    mean, std = compute_mean_std(build_split("cifar10", cifar10_dir, train=True))

    assert mean.shape == (3, 1, 1)
    assert std.shape == (3, 1, 1)
    # Batches 1..5 are filled with the constant i, ten images in all.
    expected = float(np.mean([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]))
    assert mean.flatten().tolist() == pytest.approx([expected] * 3)


def test_statistics_of_a_single_channel_dataset_have_one_entry(mnist_dir: Path) -> None:
    """The CIFAR-only version hard-coded three channels and broke here."""
    mean, std = compute_mean_std(build_split("mnist", mnist_dir, train=True))

    assert mean.shape == (1, 1, 1)
    assert std.shape == (1, 1, 1)


def test_a_constant_dataset_gets_a_nonzero_std(tmp_path: Path) -> None:
    """Dividing by a zero std would put inf into the first batch."""
    (tmp_path / "train-images-idx3-ubyte").write_bytes(
        _idx_bytes(np.full((3, 28, 28), 7, dtype=np.uint8))
    )
    (tmp_path / "train-labels-idx1-ubyte").write_bytes(_idx_bytes(np.zeros(3, dtype=np.uint8)))

    _, std = compute_mean_std(build_split("mnist", tmp_path, train=True))

    assert std.item() == 1.0


def test_the_dataset_reports_its_length(cifar10_dir: Path) -> None:
    assert len(build_split("cifar10", cifar10_dir, train=True)) == 10


def test_a_directory_named_like_the_file_is_skipped(tmp_path: Path) -> None:
    """Some MNIST copies keep each file inside a folder of the same name, next
    to the dotted files. The folder must not shadow the readable file."""
    for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"):
        (tmp_path / name).mkdir()
    (tmp_path / "train-images.idx3-ubyte").write_bytes(
        _idx_bytes(np.zeros((2, 28, 28), dtype=np.uint8))
    )
    (tmp_path / "train-labels.idx1-ubyte").write_bytes(_idx_bytes(np.array([8, 9], dtype=np.uint8)))

    _, labels = load_split("mnist", tmp_path, train=True)

    assert labels.tolist() == [8, 9]


def test_a_directory_named_like_a_cifar_batch_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "test_batch").mkdir()

    with pytest.raises(FileNotFoundError, match="test_batch"):
        load_split("cifar10", tmp_path, train=False)


# --- Covertype: one CSV, no images at all ------------------------------------


def _covertype_rows(n: int, first_label: int = 1) -> str:
    """``n`` rows of 54 features and a 1..7 label, as the UCI file writes them."""
    rows = []
    for i in range(n):
        features = [str((i + j) % 100) for j in range(54)]
        rows.append(",".join([*features, str((first_label + i - 1) % 7 + 1)]))
    return "\n".join(rows) + "\n"


def test_covertype_reads_the_csv_and_shifts_the_labels_to_zero_based(
    tmp_path: Path,
) -> None:
    """The file labels the seven cover types 1..7; cross-entropy wants 0..6."""
    (tmp_path / "covtype.data").write_text(_covertype_rows(6))
    spec = replace(spec_for("covertype"), train_rows=4)

    images, labels = spec.read(spec, tmp_path, True)

    assert images.shape == (4, 54)
    assert labels.tolist() == [0, 1, 2, 3]


def test_covertype_test_split_is_everything_after_the_training_rows(
    tmp_path: Path,
) -> None:
    (tmp_path / "covtype.data").write_text(_covertype_rows(6))
    spec = replace(spec_for("covertype"), train_rows=4)

    images, labels = spec.read(spec, tmp_path, False)

    assert images.shape == (2, 54)
    assert labels.tolist() == [4, 5]


def test_covertype_keeps_the_published_split_sizes() -> None:
    """The UCI protocol fits on the first 15,120 rows and tests on the
    remaining 565,892, so the test set here is the published one."""
    assert spec_for("covertype").train_rows == 15_120


def test_covertype_is_read_gzipped_as_downloaded(tmp_path: Path) -> None:
    with gzip.open(tmp_path / "covtype.data.gz", "wt") as f:
        f.write(_covertype_rows(6))
    spec = replace(spec_for("covertype"), train_rows=4)

    _, labels = spec.read(spec, tmp_path, True)

    assert labels.tolist() == [0, 1, 2, 3]


def test_a_covertype_file_with_the_wrong_column_count_is_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "covtype.data").write_text("1,2,3\n4,5,6\n")

    with pytest.raises(ValueError, match="expected 55"):
        load_split("covertype", tmp_path, train=True)


def test_a_covertype_file_shorter_than_the_training_split_is_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "covtype.data").write_text(_covertype_rows(3))

    with pytest.raises(ValueError, match="15,120"):
        load_split("covertype", tmp_path, train=True)


def test_a_missing_covertype_file_names_what_to_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="covtype.data"):
        load_split("covertype", tmp_path, train=True)


def test_statistics_of_a_tabular_dataset_are_one_number_per_feature(
    tmp_path: Path,
) -> None:
    """Images pool over height and width; a feature vector has neither, and the
    CIFAR-shaped code would have reduced the wrong axes."""
    (tmp_path / "covtype.data").write_text(_covertype_rows(6))
    spec = replace(spec_for("covertype"), train_rows=4)
    samples, labels = spec.read(spec, tmp_path, True)

    mean, std = compute_mean_std(ArrayDataset(samples, labels))

    assert mean.shape == (54,)
    assert std.shape == (54,)


def test_a_tabular_sample_normalizes_against_its_own_features(tmp_path: Path) -> None:
    (tmp_path / "covtype.data").write_text(_covertype_rows(6))
    spec = replace(spec_for("covertype"), train_rows=4)
    samples, labels = spec.read(spec, tmp_path, True)
    mean = torch.full((54,), 2.0)
    std = torch.full((54,), 4.0)

    plain = ArrayDataset(samples, labels)
    scaled = ArrayDataset(samples, labels, mean=mean, std=std)

    assert scaled[0][0].shape == (54,)
    assert torch.allclose(scaled[0][0], (plain[0][0] - 2.0) / 4.0)


def test_indicator_features_keep_their_own_scale() -> None:
    """Covertype's wilderness and soil columns are 0/1 flags. Z-scoring a flag
    that is set in a handful of rows turns it into a value in the hundreds, so a
    flag is left alone -- mean 0, std 1 -- while a measurement is z-scored."""
    samples = np.array([[0.0, 10.0], [1.0, 20.0], [0.0, 30.0], [1.0, 40.0]], dtype=np.float32)
    labels = np.zeros(4, dtype=np.int64)

    mean, std = compute_mean_std(ArrayDataset(samples, labels))

    assert mean.tolist() == pytest.approx([0.0, 25.0])
    assert std.tolist() == pytest.approx([1.0, float(np.std([10.0, 20.0, 30.0, 40.0]))])


def test_a_feature_constant_over_the_split_stays_bounded_elsewhere() -> None:
    """Two soil types never occur in Covertype's 15,120 training rows. With a
    floor of 1e-8 on the std, a test row that has one set arrived as 1e8 and
    the test loss came out in the hundreds; with scale 1 it arrives as 1."""
    samples = np.array([[7.0, 1.0], [7.0, 3.0]], dtype=np.float32)
    labels = np.zeros(2, dtype=np.int64)

    mean, std = compute_mean_std(ArrayDataset(samples, labels))
    row = np.array([[8.0, 2.0]], dtype=np.float32)
    elsewhere = ArrayDataset(row, labels[:1], mean=mean, std=std)
    sample, _ = elsewhere[0]

    assert (mean[0].item(), std[0].item()) == (7.0, 1.0)
    assert sample.tolist() == pytest.approx([1.0, 0.0])

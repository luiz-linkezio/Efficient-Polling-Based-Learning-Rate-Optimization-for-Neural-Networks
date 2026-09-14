"""The datasets the comparison runs on, read from the files their authors publish.

No torchvision and no download hidden inside a run: a sweep depends on numpy,
torch and the bytes already on disk. Adding a dataset is one entry in
:data:`DATASETS`, plus a reader only if its files are in a format nothing here
parses yet.

CIFAR-10 is the paper's original benchmark. The others exist to answer the
obvious objection to a single-dataset result. MNIST and Fashion-MNIST trade
three channels for one and move the difficulty in both directions; CIFAR-100
keeps the images and multiplies the classes by ten, which also moves the
random-guess loss that the divergence guard is calibrated against -- see
:func:`blowup_loss`, where a threshold left at ten classes would fire on every
batch of a hundred-class run. Covertype leaves images behind entirely: 54
tabular features and no spatial axes, so what trains there is a dense network
and the claim stops being about convolutions.

Get the files:

    # CIFAR-10 / CIFAR-100 (the *python* versions)
    curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    curl -O https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    tar -xzf cifar-10-python.tar.gz

    # MNIST / Fashion-MNIST: the four .gz files of each, in a directory of
    # their own. They are read compressed, so there is nothing to unpack.

    # Covertype: one file, read compressed as well.
    curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
"""

from __future__ import annotations

import gzip
import math
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "DATASETS",
    "ArrayDataset",
    "DatasetSpec",
    "blowup_loss",
    "build_split",
    "compute_mean_std",
    "load_split",
    "spec_for",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the rest of the code needs to know about one dataset.

    ``num_classes`` and ``in_channels`` are what the model and the divergence
    guard are built from, so they belong to the dataset rather than to a flag
    someone has to remember to pass.
    """

    key: str
    name: str
    num_classes: int
    # (C, H, W) for an image, (F,) for a feature vector. The length is what
    # decides which network the sweep builds.
    input_shape: tuple[int, ...]
    directory_hint: str
    source: str
    train_files: tuple[str, ...]
    test_files: tuple[str, ...]
    read: Callable[[DatasetSpec, Path, bool], tuple[np.ndarray, np.ndarray]] = field(
        repr=False, compare=False
    )
    # CIFAR only: which key of the pickled batch holds the labels.
    label_key: bytes | None = None
    # Covertype only: how many leading rows of the single file are the training
    # split, the rest being the test set.
    train_rows: int | None = None

    def files_for(self, train: bool) -> tuple[str, ...]:
        return self.train_files if train else self.test_files

    @property
    def is_image(self) -> bool:
        return len(self.input_shape) == 3

    @property
    def in_channels(self) -> int:
        if not self.is_image:
            raise ValueError(f"{self.name} is not an image dataset")
        return self.input_shape[0]

    @property
    def image_size(self) -> int:
        if not self.is_image:
            raise ValueError(f"{self.name} is not an image dataset")
        return self.input_shape[1]

    @property
    def figure_suffix(self) -> str:
        """What this dataset's figures append to a file name.

        CIFAR-10 appends nothing: the paper's LaTeX includes those images by
        name, and a later sweep on another dataset must not overwrite them.
        """
        return "" if self.key == "cifar10" else f"_{self.key}"


# --- CIFAR: pickled batches --------------------------------------------------


def _read_cifar(spec: DatasetSpec, data_dir: Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    paths = [data_dir / name for name in spec.files_for(train)]
    # is_file(), not exists(): some copies keep each file inside a folder of
    # the same name, and a folder is not something to hand to pickle.
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{spec.name}: missing {', '.join(missing)} in {data_dir}. "
            f"Point --data-dir at the extracted {spec.directory_hint} directory "
            f"({spec.source})."
        )

    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    side = spec.image_size
    for path in paths:
        with path.open("rb") as f:
            batch = pickle.load(f, encoding="bytes")
        if spec.label_key not in batch:
            key = (spec.label_key or b"").decode()
            raise ValueError(f"{path.name}: no {key!r} key; is this a {spec.name} batch?")
        flat = np.asarray(batch[b"data"], dtype=np.uint8)
        images.append(flat.reshape(-1, spec.in_channels, side, side))
        labels.append(np.asarray(batch[spec.label_key], dtype=np.int64))
    return np.concatenate(images), np.concatenate(labels)


# --- MNIST family: IDX files -------------------------------------------------


def _resolve(spec: DatasetSpec, data_dir: Path, name: str) -> Path:
    """Find one file, compressed or not. IDX files go by two more names in the
    wild -- dashed or dotted before ``idx`` -- so accept all four rather than
    make people rename what they downloaded."""
    candidates = [name, f"{name}.gz"]
    if "-idx" in name:
        dotted = name.replace("-idx", ".idx", 1)
        candidates += [dotted, f"{dotted}.gz"]
    for candidate in candidates:
        path = data_dir / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{spec.name}: no {name} in {data_dir} (also tried {', '.join(candidates[1:])}). "
        f"Point --data-dir at the directory holding {spec.directory_hint} ({spec.source})."
    )


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        raw = f.read()

    if len(raw) < 4:
        raise ValueError(f"{path.name}: too short to hold an IDX magic number")
    magic = int.from_bytes(raw[:4], "big")
    # 0x0000 08 nn: two zero bytes, the unsigned-byte type code, then the rank.
    if magic >> 16 != 0 or (magic >> 8) & 0xFF != 0x08:
        raise ValueError(
            f"{path.name}: bad IDX magic number 0x{magic:08x}; expected an unsigned-byte IDX file"
        )

    n_dims = magic & 0xFF
    header = 4 + 4 * n_dims
    if len(raw) < header:
        raise ValueError(f"{path.name}: truncated IDX header, {len(raw)} bytes for {n_dims} dims")
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(n_dims)]

    payload = np.frombuffer(raw, dtype=np.uint8, offset=header).copy()
    expected = int(np.prod(dims)) if dims else 0
    if payload.size != expected:
        shape = "x".join(str(d) for d in dims)
        raise ValueError(
            f"{path.name}: truncated IDX file, {payload.size} bytes of data "
            f"for the {shape} it declares ({expected} expected)"
        )
    return payload.reshape(dims)


def _read_idx_pair(spec: DatasetSpec, data_dir: Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    image_file, label_file = spec.files_for(train)
    images = _read_idx(_resolve(spec, data_dir, image_file))
    labels = _read_idx(_resolve(spec, data_dir, label_file))

    side = spec.image_size
    if images.ndim != 3:
        raise ValueError(f"{spec.name}: expected a rank-3 image file, got rank {images.ndim}")
    if images.shape[1:] != (side, side):
        got = "x".join(str(d) for d in images.shape[1:])
        raise ValueError(f"{spec.name}: images are {got}, expected {side}x{side}")
    if len(images) != len(labels):
        raise ValueError(f"{spec.name}: {len(images)} images but {len(labels)} labels")

    return images[:, None, :, :], labels.astype(np.int64)


# --- Covertype: one comma-separated table ------------------------------------


def _read_covertype(
    spec: DatasetSpec, data_dir: Path, train: bool
) -> tuple[np.ndarray, np.ndarray]:
    (name,) = spec.files_for(train)
    path = _resolve(spec, data_dir, name)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        table = np.loadtxt(f, delimiter=",", dtype=np.float32)
    table = np.atleast_2d(table)

    columns = spec.input_shape[0] + 1
    if table.shape[1] != columns:
        raise ValueError(
            f"{spec.name}: {path.name} has {table.shape[1]} columns, "
            f"expected {columns} (54 features and the cover type)"
        )

    split = spec.train_rows or 0
    if len(table) <= split:
        raise ValueError(
            f"{spec.name}: {path.name} has {len(table):,} rows, "
            f"too few for the published split of {split:,} training rows"
        )

    rows = table[:split] if train else table[split:]
    # The file numbers the seven cover types 1..7; cross-entropy wants 0..6.
    return rows[:, :-1], rows[:, -1].astype(np.int64) - 1


DATASETS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        key="cifar10",
        name="CIFAR-10",
        num_classes=10,
        input_shape=(3, 32, 32),
        directory_hint="cifar-10-batches-py",
        source="https://www.cs.toronto.edu/~kriz/cifar.html",
        train_files=tuple(f"data_batch_{i}" for i in range(1, 6)),
        test_files=("test_batch",),
        read=_read_cifar,
        label_key=b"labels",
    ),
    "cifar100": DatasetSpec(
        key="cifar100",
        name="CIFAR-100",
        num_classes=100,
        input_shape=(3, 32, 32),
        directory_hint="cifar-100-python",
        source="https://www.cs.toronto.edu/~kriz/cifar.html",
        train_files=("train",),
        test_files=("test",),
        read=_read_cifar,
        # The coarse labels are a 20-class relabelling of the same images;
        # taking them by accident would quietly become a different benchmark.
        label_key=b"fine_labels",
    ),
    "mnist": DatasetSpec(
        key="mnist",
        name="MNIST",
        num_classes=10,
        input_shape=(1, 28, 28),
        directory_hint="MNIST",
        source="https://ossci-datasets.s3.amazonaws.com/mnist/",
        train_files=("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        test_files=("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        read=_read_idx_pair,
    ),
    "fashion_mnist": DatasetSpec(
        key="fashion_mnist",
        name="Fashion-MNIST",
        num_classes=10,
        input_shape=(1, 28, 28),
        directory_hint="fashion",
        source="https://github.com/zalandoresearch/fashion-mnist",
        train_files=("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
        test_files=("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        read=_read_idx_pair,
    ),
    "covertype": DatasetSpec(
        key="covertype",
        name="Covertype",
        num_classes=7,
        input_shape=(54,),
        directory_hint="covtype.data (or covtype.data.gz)",
        source="https://archive.ics.uci.edu/dataset/31/covertype",
        train_files=("covtype.data",),
        test_files=("covtype.data",),
        read=_read_covertype,
        # The published protocol keeps the first 15,120 rows for fitting
        # (11,340 train + 3,780 validation) and the remaining 565,892 for test.
        # The test set is that one; the sweep then draws its own 90/10 split out
        # of the 15,120, the same rule it applies to every other dataset.
        train_rows=15_120,
    ),
}


def spec_for(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError:
        known = ", ".join(DATASETS)
        raise ValueError(f"unknown dataset {key!r}; known datasets are {known}") from None


def blowup_loss(spec: DatasetSpec) -> float:
    """Twice the cross-entropy of a uniform guess -- the bar above which a run
    counts as diverged.

    It has to follow the class count: ``2 * ln(10)`` is a generous ceiling on
    ten classes and *below* the ``ln(100)`` a hundred-class run starts at, so
    reusing CIFAR-10's number would roll back every batch of CIFAR-100.
    """
    return 2.0 * math.log(spec.num_classes)


def load_split(key: str, data_dir: Path | str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    """One split as ``(N, *input_shape)`` samples and int64 labels."""
    spec = spec_for(key)
    samples, labels = spec.read(spec, Path(data_dir), train)

    if samples.shape[1:] != spec.input_shape:
        got = "x".join(str(d) for d in samples.shape[1:])
        want = "x".join(str(d) for d in spec.input_shape)
        raise ValueError(f"{spec.name}: samples are {got}, expected {want}")
    if len(labels) and (labels.min() < 0 or labels.max() >= spec.num_classes):
        raise ValueError(
            f"{spec.name}: labels run {labels.min()}..{labels.max()}, "
            f"outside the {spec.num_classes} classes it declares"
        )
    return samples, labels


class ArrayDataset(Dataset):
    """Samples held in the dtype they were read in, handed out as floats.

    Pixels stay on the 0..255 scale the files use, and Covertype's features stay
    on theirs; ``mean`` and ``std`` are measured on that same scale, so the
    normalized samples are what they always were and only the memory footprint
    changed.
    """

    def __init__(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        if (mean is None) ^ (std is None):
            raise ValueError("pass `mean` and `std` together, or neither")
        self.samples = torch.from_numpy(np.ascontiguousarray(samples))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels))
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index].float()
        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
        return sample, self.labels[index]


def build_split(
    key: str,
    data_dir: Path | str,
    train: bool,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> ArrayDataset:
    samples, labels = load_split(key, data_dir, train)
    return ArrayDataset(samples, labels, mean=mean, std=std)


def compute_mean_std(dataset: Dataset, batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation along the first axis of a sample, shaped to
    broadcast back over it.

    That axis is the channel of an image, pooled over height and width, and the
    feature of a table, which has nothing to pool over -- one statistic per
    channel and per feature respectively, from the same code. A 0/1 flag is
    reported as mean 0 and std 1, which leaves it as it is.
    """
    totals: torch.Tensor | None = None
    squares: torch.Tensor | None = None
    non_flags: torch.Tensor | None = None
    sample_dims = 0
    n_values = 0

    for samples, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        samples = samples.double()
        # Everything but axis 1: the batch, and any spatial axes there may be.
        reduced = (0, *range(2, samples.ndim))
        if totals is None:
            totals = torch.zeros(samples.shape[1], dtype=torch.float64)
            squares = torch.zeros(samples.shape[1], dtype=torch.float64)
            non_flags = torch.zeros(samples.shape[1], dtype=torch.float64)
            sample_dims = samples.ndim - 1
        totals += samples.sum(dim=reduced)
        squares += samples.square().sum(dim=reduced)
        non_flags += ((samples != 0) & (samples != 1)).sum(dim=reduced)
        n_values += samples.numel() // samples.shape[1]

    if totals is None or squares is None or non_flags is None:
        raise ValueError("cannot compute statistics of an empty dataset")

    mean = totals / n_values
    std = (squares / n_values - mean.square()).clamp_min(0).sqrt()
    # A 0/1 flag (Covertype's wilderness and soil columns) keeps its own scale:
    # z-scoring a flag that is set in a handful of rows would turn it into a
    # value in the hundreds. A feature constant over the split gets scale 1, so
    # a row elsewhere that differs arrives bounded instead of divided by zero.
    # Neither case arises for the channel of an image.
    is_flag = non_flags == 0
    mean = torch.where(is_flag, torch.zeros_like(mean), mean)
    std = torch.where(is_flag | (std == 0), torch.ones_like(std), std)
    shape = (-1, *([1] * (sample_dims - 1)))
    return mean.float().view(shape), std.float().view(shape)

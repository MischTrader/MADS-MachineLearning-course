from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


# CIFAR-10 mean/std (per channel) voor Normalize.
# Deze waarden zijn standaard in veel CIFAR pipelines.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class DataLoaderConfig:
    """Config voor ImageFolder-based CIFAR subset (cifar3)."""

    data_dir: Path = Path("data/processed/cifar3")
    batch_size: int = 64
    num_workers: int = 2

    # Als je ooit wil resizen (meestal niet nodig voor CIFAR 32x32)
    resize_to: Optional[int] = None  # bijv. 32 of 64

    # Reproducibility hooks (handig voor later; ImageFolder splits zijn hier al vast)
    seed: int = 42

    # Normalization
    normalize: bool = True


def _build_transforms(cfg: DataLoaderConfig) -> Compose:
    tfs = []
    if cfg.resize_to is not None:
        tfs.append(Resize((cfg.resize_to, cfg.resize_to)))
    tfs.append(ToTensor())
    if cfg.normalize:
        tfs.append(Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD))
    return Compose(tfs)


def _assert_expected_structure(data_dir: Path) -> None:
    """Check of we een ImageFolder layout hebben met train/val subfolders."""
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir.resolve()}")

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Expected train folder not found: {train_dir.resolve()}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Expected val folder not found: {val_dir.resolve()}")

    # ImageFolder verwacht class-subfolders (cat/dog/horse)
    train_classes = [p for p in train_dir.iterdir() if p.is_dir()]
    val_classes = [p for p in val_dir.iterdir() if p.is_dir()]
    if len(train_classes) == 0:
        raise RuntimeError(f"No class folders found in: {train_dir.resolve()}")
    if len(val_classes) == 0:
        raise RuntimeError(f"No class folders found in: {val_dir.resolve()}")


def get_dataloaders(
    cfg: DataLoaderConfig,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Return: (train_loader, val_loader, class_names)

    Uses ImageFolder:
      data_dir/train/<class>/*.png
      data_dir/val/<class>/*.png
    """
    _assert_expected_structure(cfg.data_dir)

    # Determinisme: zorgt dat shuffle-volgorde reproduceerbaar is per run
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    tfm = _build_transforms(cfg)

    train_ds = ImageFolder(root=str(cfg.data_dir / "train"), transform=tfm)
    val_ds = ImageFolder(root=str(cfg.data_dir / "val"), transform=tfm)

    # Extra sanity check: classes moeten gelijk zijn
    if train_ds.classes != val_ds.classes:
        raise RuntimeError(
            "Train/val classes mismatch. "
            f"train={train_ds.classes}, val={val_ds.classes}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_ds.classes


def main() -> None:
    """Kleine smoke test: check dat loaders werken en shapes kloppen."""
    cfg = DataLoaderConfig()
    train_loader, val_loader, classes = get_dataloaders(cfg)

    x, y = next(iter(train_loader))
    print("classes:", classes)
    print("train batch:", x.shape, y.shape, "y unique:", y.unique().tolist())

    x2, y2 = next(iter(val_loader))
    print("val batch:", x2.shape, y2.shape)


if __name__ == "__main__":
    main()
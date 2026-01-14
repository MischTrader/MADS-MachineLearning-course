from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    """Configureerbare CNN-architectuur knobs voor hypertuning."""

    num_classes: int = 3  # cifar3: cat/dog/horse

    # Architecture capacity
    n_conv: int = 2  # 2..4
    base_channels: int = 32  # 16/32/48/64
    fc_units: int = 128  # 64/128/256

    # Regularization toggles
    dropout_on: bool = False
    dropout_p: float = 0.2  # alleen relevant als dropout_on=True

    batchnorm_on: bool = False

    # Skip connections (simpel residual binnen blocks)
    skip_on: bool = False

    # Activatie (optioneel tunen; default vast houden in fase 1)
    activation: str = "relu"  # "relu" | "gelu" (later eventueel)


@dataclass(frozen=True)
class TrainConfig:
    """Training knobs voor hypertuning en reproduceerbaarheid."""

    # Data
    data_dir: Path = Path("data/processed/cifar3")
    batch_size: int = 64
    num_workers: int = 2
    seed: int = 42

    # Preprocessing
    resize_to: Optional[int] = None  # None voor CIFAR 32x32
    normalize: bool = True

    # Optimizer / learning
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # "adamw" (default), later evt "adam"

    # Runtime / hardware
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    max_grad_norm: Optional[float] = None  # bv 1.0 als je instabiliteit ziet

    # Output
    results_root: Path = Path("results/ex4_Ray")

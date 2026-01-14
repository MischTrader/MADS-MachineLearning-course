from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src_ray.configs import ModelConfig, TrainConfig
from src_ray.data import DataLoaderConfig, get_dataloaders
from src_ray.model import ConfigurableCNN, count_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(x.size(0))

    if total == 0:
        raise RuntimeError("Empty evaluation loader.")
    return total_loss / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float | None = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)

        if torch.isnan(loss):
            raise RuntimeError("NaN loss encountered during training.")

        loss.backward()

        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(x.size(0))

    if total == 0:
        raise RuntimeError("Empty training loader.")
    return total_loss / total, correct / total


def _build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def run_train_eval(
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
) -> Dict[str, Any]:
    """
    Train for `train_cfg.epochs` and return a metrics dict.

    Ray-friendly: returns pure python scalars + config echo.
    """
    set_seed(train_cfg.seed)
    device = get_device(train_cfg.device)

    # Data loaders from our ImageFolder-based cifar3 setup
    dl_cfg = DataLoaderConfig(
        data_dir=train_cfg.data_dir,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        resize_to=train_cfg.resize_to,
        normalize=train_cfg.normalize,
        seed=train_cfg.seed,
    )
    train_loader, val_loader, class_names = get_dataloaders(dl_cfg)

    # Ensure model config matches dataset
    if model_cfg.num_classes != len(class_names):
        raise ValueError(
            f"ModelConfig.num_classes ({model_cfg.num_classes}) does not match "
            f"dataset classes ({len(class_names)}): {class_names}"
        )

    model = ConfigurableCNN(model_cfg).to(device)
    n_params = count_params(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(train_cfg, model)

    best_val_acc = 0.0
    best_val_loss = float("inf")

    history = []  # optional: keep per-epoch metrics for debugging
    t0 = time.time()

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            device,
            loss_fn,
            optimizer,
            max_grad_norm=train_cfg.max_grad_norm,
        )
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)

        best_val_acc = max(best_val_acc, val_acc)
        best_val_loss = min(best_val_loss, val_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    wall_time_s = time.time() - t0

    # Return a compact summary + echoes for traceability
    return {
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "num_params": int(n_params),
        "wall_time_s": float(wall_time_s),
        "class_names": list(class_names),
        "train_cfg": asdict(train_cfg),
        "model_cfg": asdict(model_cfg),
        "history": history,  # keep for now; we can drop later for speed/size
    }


def main() -> None:
    """Quick smoke test run."""
    train_cfg = TrainConfig(epochs=2, device="cpu")
    model_cfg = ModelConfig(
        num_classes=3,
        n_conv=2,
        base_channels=32,
        fc_units=128,
        dropout_on=False,
        batchnorm_on=False,
        skip_on=False,
    )
    res = run_train_eval(train_cfg, model_cfg)
    print("best_val_acc:", res["best_val_acc"])
    print("best_val_loss:", res["best_val_loss"])
    print("num_params:", res["num_params"])


if __name__ == "__main__":
    main()

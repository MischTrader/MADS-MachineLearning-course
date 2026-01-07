from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

import mlflow
import mlflow.pytorch

from src.model import ModelConfig, SmallCNN


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path("data/processed/cifar3")
    batch_size: int = 64
    epochs: int = 5
    num_workers: int = 2
    seed: int = 42

    # hyperparameters
    lr: float = 1e-3
    weight_decay: float = 0.0

    # model knobs
    base_channels: int = 16
    fc_units: int = 64
    dropout: float = 0.0
    norm: str | None = None

    # logging
    log_root: Path = Path("runs/grid")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def train_one_epoch(model, loader, device, loss_fn, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def run_training(cfg: TrainConfig, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = Compose([ToTensor()])
    train_ds = ImageFolder(cfg.data_dir / "train", transform=tfm)
    val_ds = ImageFolder(cfg.data_dir / "val", transform=tfm)

    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    mcfg = ModelConfig(
        num_classes=len(train_ds.classes),
        base_channels=cfg.base_channels,
        fc_units=cfg.fc_units,
        dropout=cfg.dropout,
        norm=cfg.norm,          # <-- TOEGEVOEGD
    )
    model = SmallCNN(mcfg).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if run_name is None:
        run_name = f"run_{int(time.time())}"

    # ---------- MLflow ----------
    mlflow.set_experiment("exercise_2")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("device", str(device))

        # TensorBoard (optioneel)
        log_dir = cfg.log_root / run_name
        writer = SummaryWriter(log_dir=str(log_dir))

        best_val_acc = 0.0
        best_val_loss = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, device, loss_fn, optimizer
            )
            val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.pytorch.log_model(model, artifact_path="model")

        writer.close()

    return {
        "run_name": run_name,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        **asdict(cfg),
    }


def main() -> None:
    cfg = TrainConfig(epochs=2, norm="batchnorm")
    res = run_training(cfg, run_name="single_run_check")
    print(res)

if __name__ == "__main__":
    main()
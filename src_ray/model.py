from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from src_ray.configs import ModelConfig


def count_params(model: nn.Module) -> int:
    """Aantal trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """
    Conv -> (BN?) -> Act -> (Dropout?) block.
    Optioneel: residual/skip connection als in/out channels matchen.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        batchnorm_on: bool,
        dropout_on: bool,
        dropout_p: float,
        activation: str,
        skip_on: bool,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not batchnorm_on)
        self.bn = nn.BatchNorm2d(out_ch) if batchnorm_on else nn.Identity()
        self.act = _get_activation(activation)

        # Dropout2d is vaak logischer in conv stacks dan gewone Dropout
        self.drop = nn.Dropout2d(p=dropout_p) if dropout_on else nn.Identity()

        # Skip is alleen "simpel" toegestaan als shapes matchen
        self.use_skip = bool(skip_on and in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        if self.use_skip:
            out = out + x
        return out


class ConfigurableCNN(nn.Module):
    """
    Configureerbare CNN voor 32x32 RGB (cifar3).
    - n_conv blocks (2..4)
    - channels groeien per block
    - pooling om spatial size te verkleinen
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if not (2 <= cfg.n_conv <= 4):
            raise ValueError(f"n_conv must be in [2,4], got {cfg.n_conv}")

        # channels: base, 2*base, 4*base, 8*base (afhankelijk van n_conv)
        channels = [cfg.base_channels * (2**i) for i in range(cfg.n_conv)]
        blocks: list[nn.Module] = []

        in_ch = 3
        for i, out_ch in enumerate(channels):
            blocks.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    batchnorm_on=cfg.batchnorm_on,
                    dropout_on=cfg.dropout_on,
                    dropout_p=cfg.dropout_p,
                    activation=cfg.activation,
                    skip_on=cfg.skip_on,
                )
            )

            # Pooling na elk block behalve eventueel het laatste (simpel en voorspelbaar)
            # Voor 32x32: 2 pools -> 8x8, 3 pools -> 4x4, 4 pools -> 2x2
            if i < cfg.n_conv - 1:
                blocks.append(nn.MaxPool2d(kernel_size=2))

            in_ch = out_ch

        self.features = nn.Sequential(*blocks)

        # We bepalen flatten-dim op een robuuste manier met een dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feat = self.features(dummy)
            flat_dim = int(feat.flatten(1).shape[1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, cfg.fc_units),
            nn.BatchNorm1d(cfg.fc_units) if cfg.batchnorm_on else nn.Identity(),
            _get_activation(cfg.activation),
            nn.Dropout(p=cfg.dropout_p) if cfg.dropout_on else nn.Identity(),
            nn.Linear(cfg.fc_units, cfg.num_classes),
        )

        # Handig voor debug/trace
        self.cfg_dict = asdict(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def main() -> None:
    """Smoke test: bouw model, forward, param count."""
    cfg = ModelConfig(
        num_classes=3,
        n_conv=3,
        base_channels=32,
        fc_units=128,
        dropout_on=True,
        dropout_p=0.2,
        batchnorm_on=True,
        skip_on=True,
        activation="relu",
    )
    model = ConfigurableCNN(cfg)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print("output:", y.shape)
    print("params:", count_params(model))


if __name__ == "__main__":
    main()

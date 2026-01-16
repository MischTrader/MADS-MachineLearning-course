# src_ray/plot_phase2.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EXPORT_CSV = Path("results/ex4_Ray/exports/phase2.csv")
PLOTS_DIR = Path("results/ex4_Ray/plots")


def _load_phase2() -> pd.DataFrame:
    if not EXPORT_CSV.exists():
        raise FileNotFoundError(f"Missing: {EXPORT_CSV.resolve()}")

    df = pd.read_csv(EXPORT_CSV)

    # Keep only successful trials if column exists
    if "status" in df.columns:
        df = df[df["status"] == "TERMINATED"].copy()

    # Basic sanity
    if df.empty:
        raise RuntimeError("phase2.csv loaded but contains no successful trials.")

    return df


def plot1_heatmap_lr_vs_channels(df: pd.DataFrame) -> None:
    """
    Plot 1: Heatmap of lr (binned on log scale) x base_channels -> mean val_acc,
    faceted by skip_on (False/True).
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Extract columns
    lr = df["config/lr"].astype(float).to_numpy()
    channels = df["config/base_channels"].astype(int)
    val_acc = df["val_acc"].astype(float)

    # Log-spaced bins for lr
    log_lr = np.log10(lr)
    n_bins = 6
    bins = np.linspace(log_lr.min(), log_lr.max(), n_bins + 1)
    df = df.copy()
    df["lr_bin"] = pd.cut(log_lr, bins=bins, include_lowest=True)

    # We'll facet by skip_on if present, else just one plot
    if "config/skip_on" in df.columns:
        facet_values = [False, True]
    else:
        facet_values = [None]

    for skip_val in facet_values:
        if skip_val is None:
            df_sub = df
            suffix = "all"
            title_suffix = ""
        else:
            df_sub = df[df["config/skip_on"] == skip_val]
            suffix = f"skip_{skip_val}".lower()
            title_suffix = f" (skip_on={skip_val})"

        if df_sub.empty:
            continue

        # Pivot: rows=lr_bin, cols=base_channels
        pivot = (
            df_sub.pivot_table(
                index="lr_bin",
                columns="config/base_channels",
                values="val_acc",
                aggfunc="mean",
            )
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_title(f"Phase 2: val_acc heatmap lr_bin × base_channels{title_suffix}")
        ax.set_xlabel("base_channels")
        ax.set_ylabel("learning rate (log-binned)")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(b) for b in pivot.index])

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("mean val_acc")

        out_path = PLOTS_DIR / f"phase2_heatmap_lr_x_channels_{suffix}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"[plot1] Wrote: {out_path.resolve()}")


def plot2_scatter_params_vs_acc(df: pd.DataFrame) -> None:
    """
    Plot 2: Scatter num_params vs val_acc, marker by skip_on if present.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    if "config/skip_on" in df.columns:
        for skip_val, marker in [(False, "o"), (True, "x")]:
            sub = df[df["config/skip_on"] == skip_val]
            if sub.empty:
                continue
            ax.scatter(
                sub["num_params"].astype(float),
                sub["val_acc"].astype(float),
                marker=marker,
                label=f"skip_on={skip_val}",
            )
        ax.legend()
    else:
        ax.scatter(df["num_params"].astype(float), df["val_acc"].astype(float))

    ax.set_title("Phase 2: num_params vs val_acc")
    ax.set_xlabel("num_params")
    ax.set_ylabel("val_acc")

    out_path = PLOTS_DIR / "phase2_scatter_params_vs_acc.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[plot2] Wrote: {out_path.resolve()}")


def main() -> None:
    df = _load_phase2()
    plot1_heatmap_lr_vs_channels(df)
    plot2_scatter_params_vs_acc(df)


if __name__ == "__main__":
    main()
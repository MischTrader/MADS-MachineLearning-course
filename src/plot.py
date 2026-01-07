from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def save_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    """Save a heatmap of best_val_acc for lr × fc_units for the provided dataframe."""
    df = df.copy()

    # Ensure consistent dtypes
    df["lr"] = df["lr"].astype(float)
    df["fc_units"] = df["fc_units"].astype(int)

    # Pivot for heatmap: rows=fc_units, cols=lr, values=best_val_acc
    pivot = (
        df.pivot_table(
            index="fc_units",
            columns="lr",
            values="best_val_acc",
            aggfunc="max",
        )
        .sort_index()
    )

    plt.figure()
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im, label="best_val_acc")

    # Axis ticks/labels
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xticks(
        range(len(pivot.columns)),
        [f"{x:.4g}" for x in pivot.columns],
        rotation=45,
    )

    plt.xlabel("learning rate (lr)")
    plt.ylabel("fc_units")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    metrics_path = Path("results/metrics.csv")
    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)

    # Split by norm so we explicitly visualize the added hyperparameter
    # Note: depending on how pandas reads None, you may see NaN or the string "None".
    df_none = df[df["norm"].isna() | (df["norm"] == "None")]
    df_bn = df[df["norm"] == "batchnorm"]

    # New filenames so Exercise 1 plot isn't overwritten
    out_none = out_dir / "ex2_heatmap_lr_fc_norm_none.png"
    out_bn = out_dir / "ex2_heatmap_lr_fc_norm_batchnorm.png"

    if len(df_none) > 0:
        save_heatmap(
            df_none,
            out_none,
            "Exercise 2: lr × fc_units (best_val_acc) | norm=None",
        )
        print(f"Saved: {out_none.resolve()}")
    else:
        print("Skipped norm=None plot: no matching rows found in metrics.csv")

    if len(df_bn) > 0:
        save_heatmap(
            df_bn,
            out_bn,
            "Exercise 2: lr × fc_units (best_val_acc) | norm=batchnorm",
        )
        print(f"Saved: {out_bn.resolve()}")
    else:
        print("Skipped norm=batchnorm plot: no matching rows found in metrics.csv")


if __name__ == "__main__":
    main()

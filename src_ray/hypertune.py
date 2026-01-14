# src_ray/hypertune.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from ray import tune
from ray.tune import TuneConfig, RunConfig
from ray.tune.schedulers import ASHAScheduler

from src_ray.configs import ModelConfig, TrainConfig
from src_ray.train import run_train_eval

def trainable(config: Dict[str, Any]) -> None:
    """
    Ray Tune trainable:
    - config -> (ModelConfig, TrainConfig)
    - traint model
    - rapporteert metrics via tune.report(...)
    """
    # --- Split config into model/train parts (simple, explicit) ---
    model_keys = set(asdict(ModelConfig()).keys())
    train_keys = set(asdict(TrainConfig()).keys())

    model_kwargs = {k: config[k] for k in config.keys() if k in model_keys}
    train_kwargs = {k: config[k] for k in config.keys() if k in train_keys}

    # --- FIX: Ray geeft data_dir als str; TrainConfig verwacht Path ---
    if "data_dir" in train_kwargs:
        train_kwargs["data_dir"] = Path(train_kwargs["data_dir"]).resolve()

    model_cfg = ModelConfig(**model_kwargs)
    train_cfg = TrainConfig(**train_kwargs)

    # --- Train/Eval ---
    res = run_train_eval(train_cfg=train_cfg, model_cfg=model_cfg,)

    # --- Report to Ray ---
    tune.report({
        "val_acc": res["best_val_acc"],
        "val_loss": res["best_val_loss"],
        "num_params": res["num_params"],
        "wall_time_s": res["wall_time_s"],
    })

def run_phase1() -> None:
    results_root = Path("results/ex4_Ray/ray_results/phase1")
    results_root.mkdir(parents=True, exist_ok=True)

    search_space: dict[str, Any] = {
        "epochs": 5,
        "batch_size": tune.choice([64, 128]),
        "lr": tune.loguniform(1e-4, 3e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "optimizer": tune.choice(["adamw"]),
        "device": "cpu",
        "seed": 42,
        "data_dir": str(Path("data/processed/cifar3").resolve()),
        "num_classes": 3,
        "n_conv": tune.choice([2, 3, 4]),
        "base_channels": tune.choice([32, 48, 64]),
        "fc_units": tune.choice([64, 128, 256]),
        "batchnorm_on": tune.choice([False, True]),
        "dropout_on": tune.choice([False, True]),
        "dropout_p": tune.choice([0.2, 0.4]),
        "skip_on": tune.choice([False, True]),
        "activation": "relu",
    }

    scheduler = ASHAScheduler(metric="val_acc", mode="max")

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=TuneConfig(
            num_samples=10,
            max_concurrent_trials=2,
            scheduler=scheduler,
        ),
        run_config=RunConfig(storage_path=str(results_root.resolve())),
    )

    tuner.fit()

def run_phase2() -> None:
    results_root = Path("results/ex4_Ray/ray_results/phase2")
    results_root.mkdir(parents=True, exist_ok=True)

    search_space: dict[str, Any] = {
        "epochs": 12,
        "batch_size": tune.choice([64, 128]),
        "lr": tune.loguniform(3e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
        "optimizer": tune.choice(["adamw"]),
        "device": "cpu",
        "seed": 42,
        "data_dir": str(Path("data/processed/cifar3").resolve()),
        "num_classes": 3,
        "n_conv": tune.choice([3]),
        "base_channels": tune.choice([48, 64]),
        "fc_units": tune.choice([128, 256]),
        "batchnorm_on": tune.choice([False, True]),
        "dropout_on": tune.choice([False, True]),
        "dropout_p": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "skip_on": tune.choice([False, True]),
        "activation": "relu",
    }

    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=TuneConfig(
            num_samples=10,
            max_concurrent_trials=2,
        ),
        run_config=RunConfig(storage_path=str(results_root.resolve())),
    )

    tuner.fit()

def main() -> None:
    # Minimal CLI switch via env var would be overkill; keep it simple:
    # Uncomment one at a time.
    run_phase1()
    # run_phase2()


if __name__ == "__main__":
    main()

"""Run and aggregate the Stage 2 nested training-size study."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .stage2_baselines import run


TRAINING_SIZES = (250, 500, 1000, 2000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", default="data/shared_effects")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--out", default="outputs/stage2_training_sizes")
    parser.add_argument("--figures", default="outputs/stage2_figures",
                        help="Directory for generated learning-curve figures")
    args = parser.parse_args()

    output_dir = Path(args.out)
    figures_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for train_size in TRAINING_SIZES:
        size_dir = output_dir / f"n{train_size}"
        run(args.data, size_dir, scores=None,
            train_size=train_size, replicate=args.replicate)
        metrics = pd.read_csv(size_dir / "validation_metrics.csv")
        metrics["train_size"] = train_size
        rows.append(metrics)

    learning_curve = pd.concat(rows, ignore_index=True)
    learning_curve.to_csv(output_dir / "learning_curve_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, group in learning_curve.groupby("method"):
        group = group.sort_values("train_size")
        axes[0].plot(group["train_size"], group["validation_incremental_r2"],
                     marker="o", label=method)
        axes[1].plot(group["train_size"], group["validation_mse"],
                     marker="o", label=method)

    axes[0].set_title("Validation incremental R2")
    axes[0].set_xlabel("Training size")
    axes[0].set_ylabel("Incremental R2")
    axes[1].set_title("Validation MSE")
    axes[1].set_xlabel("Training size")
    axes[1].set_ylabel("MSE")
    axes[1].legend(fontsize=8)

    fig.savefig(figures_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    print("Wrote training-size results to:", output_dir.resolve())
    print("Wrote training-size figure to:", (figures_dir / "learning_curves.png").resolve())


if __name__ == "__main__":
    main()

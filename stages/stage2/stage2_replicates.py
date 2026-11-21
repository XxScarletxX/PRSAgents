"""Run Stage 2 baselines across replicates and summarize paired differences."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .stage2_baselines import run


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows[rows["method"] == "stage2_moe"][["replicate", "validation_incremental_r2"]]
    baseline = baseline.rename(columns={"validation_incremental_r2": "stage2_moe_r2"})
    paired = rows.merge(baseline, on="replicate", how="inner")
    paired["paired_delta_r2"] = paired["validation_incremental_r2"] - paired["stage2_moe_r2"]

    summary = paired.groupby("method").agg(
        replicate_count=("replicate", "nunique"),
        mean_r2=("validation_incremental_r2", "mean"),
        mean_paired_delta_vs_moe=("paired_delta_r2", "mean"),
    ).reset_index()
    summary["se_paired_delta_vs_moe"] = paired.groupby("method")[
        "paired_delta_r2"
    ].sem(ddof=1).reindex(summary["method"]).to_numpy()
    summary["mean_validation_mse"] = paired.groupby("method")[
        "validation_mse"
    ].mean().reindex(summary["method"]).to_numpy()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", default="data/shared_effects")
    parser.add_argument("--replicates", default="0-9",
                        help="Replicate range in the form START-END")
    parser.add_argument("--out", default="outputs/stage2_replicates")
    parser.add_argument("--figures", default="outputs/stage2_figures")
    args = parser.parse_args()

    start_text, end_text = args.replicates.split("-", 1)
    replicates = range(int(start_text), int(end_text) + 1)

    output_dir = Path(args.out)
    figures_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for replicate in replicates:
        replicate_dir = output_dir / f"rep{replicate}"
        run(args.data, replicate_dir, scores=None,
            train_size=None, replicate=replicate)
        metrics = pd.read_csv(replicate_dir / "validation_metrics.csv")
        metrics["replicate"] = replicate
        rows.append(metrics)

    replicate_metrics = pd.concat(rows, ignore_index=True)
    replicate_metrics.to_csv(output_dir / "replicate_metrics.csv", index=False)

    summary = paired_summary(replicate_metrics)
    summary.to_csv(output_dir / "paired_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for method, group in replicate_metrics.groupby("method"):
        axes[0].scatter(group["replicate"], group["validation_incremental_r2"],
                        s=28, label=method)
        axes[1].scatter(group["replicate"], group["validation_mse"],
                        s=28, label=method)

    axes[0].set_title("Validation incremental R2 by replicate")
    axes[0].set_xlabel("Replicate")
    axes[0].set_ylabel("Incremental R2")
    axes[1].set_title("Validation MSE by replicate")
    axes[1].set_xlabel("Replicate")
    axes[1].set_ylabel("MSE")
    axes[1].legend(fontsize=8)

    fig.savefig(figures_dir / "replicate_metrics.png", dpi=160)
    plt.close(fig)

    print("Wrote replicate results to:", output_dir.resolve())
    print("Wrote replicate figure to:", (figures_dir / "replicate_metrics.png").resolve())


if __name__ == "__main__":
    main()

"""Plot Stage 2 baseline diagnostics from existing local output CSVs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_validation_metrics(input_dir: Path, output_dir: Path) -> None:
    metrics = pd.read_csv(input_dir / "validation_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(metrics["method"], metrics["validation_incremental_r2"])
    axes[0].set_title("Validation incremental R2")
    axes[0].set_ylabel("Incremental R2")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(metrics["method"], metrics["validation_mse"])
    axes[1].set_title("Validation MSE")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis="x", rotation=35)

    fig.savefig(output_dir / "validation_metrics.png", dpi=160)
    plt.close(fig)


def plot_training_likelihood(input_dir: Path, output_dir: Path) -> None:
    trace = pd.read_csv(input_dir / "moe_training_loglik.csv")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(trace["iteration"], trace["training_log_likelihood"], marker="o", ms=2)
    ax.set_title("Stage 2 training log likelihood")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Unpenalized log likelihood")

    fig.savefig(output_dir / "moe_training_likelihood.png", dpi=160)
    plt.close(fig)


def plot_gate_weights(input_dir: Path, output_dir: Path) -> None:
    gates = pd.read_csv(input_dir / "validation_gate_weights.csv")
    expert_columns = [column for column in gates.columns if column.endswith("_viprs")]
    if not expert_columns:
        raise ValueError("No *_viprs gate-weight columns found")

    weights = gates[expert_columns].to_numpy(float)
    if not np.isfinite(weights).all():
        raise ValueError("Gate weights contain non-finite values")

    row_sums = weights.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("Gate weights do not sum to one")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for pc_name, ax in (("PC1", axes[0]), ("PC2", axes[1])):
        for expert_name in expert_columns:
            ax.scatter(gates[pc_name], gates[expert_name], s=8, alpha=0.35,
                       label=expert_name.replace("_viprs", ""))
        ax.set_title(f"Gate weights by {pc_name}")
        ax.set_xlabel(pc_name)
        ax.set_ylabel("Gate weight")
        ax.set_ylim(-0.02, 1.02)

    axes[0].legend(title="Expert", fontsize=8)
    fig.savefig(output_dir / "gate_weights_by_pc.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="outputs/stage2_demo",
                        help="Directory containing Stage 2 CSV exports")
    parser.add_argument("--out", default="outputs/stage2_figures",
                        help="Directory for generated figures")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_validation_metrics(input_dir, output_dir)
    plot_training_likelihood(input_dir, output_dir)
    plot_gate_weights(input_dir, output_dir)

    print("Wrote Stage 2 diagnostics to:", output_dir.resolve())


if __name__ == "__main__":
    main()

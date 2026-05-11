from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and fail clearly if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def plot_bar(
    df: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot mean ± std bar chart from a flat summary table."""
    labels = df["model_name"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        labels,
        df[metric_mean],
        yerr=df[metric_std].fillna(0.0),
        capsize=4,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Generate final corrected ANODE figures from flat CSV tables."""
    results_dir = Path("results/anode/final")
    figures_dir = Path("figures/anode/corrected")
    figures_dir.mkdir(parents=True, exist_ok=True)

    circles = load_csv(results_dir / "circles_corrected_flat_table_job492199.csv")
    slice_df = load_csv(
        results_dir / "slice_corrected_flat_table_job492288_epochs500.csv"
    )

    circle_plots: List[Tuple[str, str, str, str, str]] = [
        (
            "final_val_accuracy_mean",
            "final_val_accuracy_std",
            "Corrected circles: validation accuracy",
            "Validation accuracy",
            "circles_corrected_val_accuracy.png",
        ),
        (
            "final_val_loss_mean",
            "final_val_loss_std",
            "Corrected circles: validation loss",
            "Validation loss",
            "circles_corrected_val_loss.png",
        ),
        (
            "final_train_total_nfe_mean",
            "final_train_total_nfe_std",
            "Corrected circles: total training NFE",
            "Forward + backward NFE",
            "circles_corrected_train_total_nfe.png",
        ),
        (
            "final_val_forward_nfe_mean",
            "final_val_forward_nfe_std",
            "Corrected circles: validation forward NFE",
            "Forward NFE",
            "circles_corrected_val_forward_nfe.png",
        ),
    ]

    for mean_col, std_col, title, ylabel, filename in circle_plots:
        plot_bar(circles, mean_col, std_col, title, ylabel, figures_dir / filename)

    slice_plots: List[Tuple[str, str, str, str, str]] = [
        (
            "final_full_val_accuracy_mean",
            "final_full_val_accuracy_std",
            "Missing-slice: full validation accuracy",
            "Full validation accuracy",
            "slice_corrected_full_val_accuracy.png",
        ),
        (
            "final_slice_val_accuracy_mean",
            "final_slice_val_accuracy_std",
            "Missing-slice: held-out slice accuracy",
            "Held-out slice accuracy",
            "slice_corrected_slice_val_accuracy.png",
        ),
        (
            "final_slice_val_loss_mean",
            "final_slice_val_loss_std",
            "Missing-slice: held-out slice loss",
            "Held-out slice loss",
            "slice_corrected_slice_val_loss.png",
        ),
        (
            "final_train_total_nfe_mean",
            "final_train_total_nfe_std",
            "Missing-slice: total training NFE",
            "Forward + backward NFE",
            "slice_corrected_train_total_nfe.png",
        ),
        (
            "final_slice_val_forward_nfe_mean",
            "final_slice_val_forward_nfe_std",
            "Missing-slice: held-out slice forward NFE",
            "Forward NFE",
            "slice_corrected_slice_forward_nfe.png",
        ),
    ]

    for mean_col, std_col, title, ylabel, filename in slice_plots:
        plot_bar(slice_df, mean_col, std_col, title, ylabel, figures_dir / filename)

    print(f"Saved corrected ANODE figures to {figures_dir}")


if __name__ == "__main__":
    main()

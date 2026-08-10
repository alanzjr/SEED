"""Time-series probability curves relative to a mainshock."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_probability_curve(
    csv_path: str | Path,
    output_path: str | Path,
    post_event_days: int = 200,
    show: bool = False,
) -> Path:
    """
    Plot mean probability with min–max envelope.

    Expects columns such as:
    Window_Index, Max_EQ_Probability(%), Min_EQ_Probability(%), Mean_EQ_Probability(%)
    or Min_Probability / Max_Probability / Avg_Probability / Mean_EQ_Probability(%).
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in data.columns}

    def _pick(*names):
        for name in names:
            if name.lower() in cols:
                return cols[name.lower()]
            # also try exact
            if name in data.columns:
                return name
        return None

    col_max = _pick("Max_EQ_Probability(%)", "Max_Probability", "max")
    col_min = _pick("Min_EQ_Probability(%)", "Min_Probability", "min")
    col_avg = _pick("Mean_EQ_Probability(%)", "Avg_Probability", "Mean_Probability", "avg")
    if col_avg is None:
        # fall back to column order used by legacy figure.py: idx, max, min, avg
        if data.shape[1] >= 4:
            col_max, col_min, col_avg = data.columns[1], data.columns[2], data.columns[3]
        else:
            raise ValueError(f"Cannot identify probability columns in {csv_path}")

    y_max = data[col_max].values.astype(float)
    y_min = data[col_min].values.astype(float)
    y_avg = data[col_avg].values.astype(float)
    # Convert 0–1 fractions to percent if needed
    if np.nanmax(y_avg) <= 1.5:
        y_max, y_min, y_avg = y_max * 100, y_min * 100, y_avg * 100

    len_data = len(data)
    x = np.linspace(-(len_data - post_event_days - 1), post_event_days, len_data)

    plt.figure(figsize=(18, 9))
    plt.plot(x, y_avg, label="Average Probability", color="#D90429", linewidth=3)
    plt.fill_between(
        x, y_min, y_max, color="#FFB3B3", alpha=0.4, label="Min-Max Probability Range"
    )
    plt.axvline(x=0, color="black", linestyle="--", linewidth=3)
    plt.xlim(-(len_data - post_event_days - 1), post_event_days)
    plt.ylim(0, 100)
    plt.xlabel("Days relative to mainshock", fontsize=40)
    plt.ylabel(r"$\mathrm{P_{un}}$ (%)", fontsize=40)
    plt.tick_params(axis="both", which="major", labelsize=30)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
    plt.legend(loc="upper left", fontsize=30)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return output_path

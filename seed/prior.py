"""Spatially varying prior (SVP) probability map from a catalog."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_svp_grid(
    catalog_path: str | Path,
    start_date: str = "2000-01-01",
    end_date: str = "2014-12-31",
    lon_min: float = 100.0,
    lon_max: float = 105.0,
    lat_min: float = 28.0,
    lat_max: float = 35.0,
    grid_size: float = 0.5,
    mag_threshold: float = 5.0,
    t_years: float = 10.0,
    window_days: float = 30.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute Poisson-based prior probability on a lon/lat grid.

    Returns
    -------
    grid_full : DataFrame with lon_center, lat_center, earthquake_count, prior_prob
    lon_bins, lat_bins : bin edges for plotting
    """
    df = pd.read_csv(catalog_path)
    df["time"] = pd.to_datetime(df["time"])
    mask = (df["time"] >= start_date) & (df["time"] <= end_date)
    df = df.loc[mask].copy()

    df_m5 = df[df["mag"] >= mag_threshold].copy()
    lon_bins = np.arange(lon_min, lon_max + grid_size, grid_size)
    lat_bins = np.arange(lat_min, lat_max + grid_size, grid_size)
    lon_centers = lon_bins[:-1] + grid_size / 2
    lat_centers = lat_bins[:-1] + grid_size / 2

    grid_full = (
        pd.MultiIndex.from_product([lon_centers, lat_centers], names=["lon_center", "lat_center"])
        .to_frame(index=False)
    )
    df_m5["lon_center"] = (
        np.floor((df_m5["longitude"] - lon_min) / grid_size) * grid_size + lon_min + grid_size / 2
    )
    df_m5["lat_center"] = (
        np.floor((df_m5["latitude"] - lat_min) / grid_size) * grid_size + lat_min + grid_size / 2
    )
    counts = (
        df_m5.groupby(["lon_center", "lat_center"]).size().reset_index(name="earthquake_count")
    )
    grid_full = pd.merge(grid_full, counts, on=["lon_center", "lat_center"], how="left").fillna(
        {"earthquake_count": 0}
    )
    grid_full["prior_prob"] = grid_full["earthquake_count"] / t_years
    grid_full["prior_prob"] = 1 - np.exp(-grid_full["prior_prob"] * (window_days / 365.0))
    grid_full["prior_prob"] = grid_full["prior_prob"].replace(0, 1e-8)
    return grid_full, lon_bins, lat_bins


def plot_svp_map(
    grid_full: pd.DataFrame,
    lon_bins: np.ndarray,
    lat_bins: np.ndarray,
    output_path: str | Path,
    mag_threshold: float = 5.0,
) -> Path:
    """Save a pcolormesh map of SVP probabilities."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    pivot_prob = grid_full.pivot(index="lat_center", columns="lon_center", values="prior_prob")
    mesh = ax.pcolormesh(
        lon_bins,
        lat_bins,
        pivot_prob.values,
        cmap="YlOrRd",
        edgecolors="w",
        linewidth=0.1,
        shading="auto",
    )
    cbar = plt.colorbar(mesh)
    cbar.set_label(
        f"Spatially Varying Prior Probability (SVP)\n(M>={mag_threshold})",
        fontsize=12,
    )
    for _, row in grid_full.iterrows():
        prob_percent = row["prior_prob"] * 100
        text_color = "white" if prob_percent > grid_full["prior_prob"].max() * 50 else "black"
        ax.text(
            row["lon_center"],
            row["lat_center"],
            f"{prob_percent:.2f}%",
            ha="center",
            va="center",
            fontsize=7,
            color=text_color,
        )
    ax.set_xlabel("Longitude (°)", fontsize=11)
    ax.set_ylabel("Latitude (°)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path

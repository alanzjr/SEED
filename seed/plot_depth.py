"""Depth-section plots from statistics_depth_* CSV grids."""

from __future__ import annotations

from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import griddata
from tqdm import tqdm


def plot_depth_sections(
    data_dir: str | Path,
    output_dir_lon: str | Path,
    output_dir_lat: str | Path,
    lat_min: float = 31.0,
    lat_max: float = 35.0,
    lon_min: float = 101.5,
    lon_max: float = 105.5,
    epicenter_lon: float = 87.45,
    epicenter_lat: float = 28.5,
    hypocenter_depth: float = 10.0,
    num_days: int = 566,
    max_days: int | None = None,
) -> None:
    """
    Produce depth-vs-longitude and depth-vs-latitude sections for each day.

    Expected filenames: statistics_depth_{depth}_lat_{lat}_lon_{lon}.csv
    with columns Window_Index and Avg_Probability.
    """
    data_dir = Path(data_dir)
    output_dir_lon = Path(output_dir_lon)
    output_dir_lat = Path(output_dir_lat)
    output_dir_lon.mkdir(parents=True, exist_ok=True)
    output_dir_lat.mkdir(parents=True, exist_ok=True)

    files = glob(str(data_dir / "statistics_depth_*.csv"))
    depths, lats, lons, valid_files = [], [], [], []
    for file in files:
        parts = Path(file).name.split("_")
        # statistics_depth_{d}_lat_{lat}_lon_{lon}.csv
        try:
            depth = float(parts[2])
            lat = float(parts[4])
            lon = float(parts[6].replace(".csv", ""))
        except (IndexError, ValueError):
            continue
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            depths.append(depth)
            lats.append(lat)
            lons.append(lon)
            valid_files.append(file)

    if not valid_files:
        raise FileNotFoundError(
            f"No depth statistics CSVs found in {data_dir}. "
            "Provide files named statistics_depth_{depth}_lat_{lat}_lon_{lon}.csv"
        )

    depths = sorted(set(depths))
    lats = sorted(set(lats))
    lons = sorted(set(lons))

    data_dict = {}
    for file in tqdm(valid_files, desc="Reading depth grids"):
        parts = Path(file).name.split("_")
        depth = float(parts[2])
        lat = float(parts[4])
        lon = float(parts[6].replace(".csv", ""))
        df = pd.read_csv(file)
        if "Window_Index" not in df.columns or "Avg_Probability" not in df.columns:
            continue
        probs = df.sort_values("Window_Index")["Avg_Probability"].values
        if len(probs) >= num_days:
            data_dict[(depth, lat, lon)] = probs[:num_days]
        else:
            data_dict[(depth, lat, lon)] = np.pad(
                probs, (0, num_days - len(probs)), constant_values=np.nan
            )

    colors = ["#b3e5fc", "#fff176", "#ff7043", "#d32f2f"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    norm = Normalize(vmin=0, vmax=100)
    days = range(num_days if max_days is None else min(num_days, max_days))

    for day in tqdm(days, desc="Plotting depth sections"):
        points_lon, values_lon = [], []
        for depth in depths:
            for lon in lons:
                vals = [
                    data_dict[(depth, lat, lon)][day]
                    for lat in lats
                    if (depth, lat, lon) in data_dict
                    and not np.isnan(data_dict[(depth, lat, lon)][day])
                ]
                if vals:
                    points_lon.append((lon, depth))
                    values_lon.append(float(np.mean(vals)))

        if len(points_lon) >= 3:
            grid_lon, grid_depth = np.meshgrid(
                np.linspace(min(lons), max(lons), 100),
                np.linspace(min(depths), max(depths), 200),
            )
            grid_values_lon = griddata(points_lon, values_lon, (grid_lon, grid_depth), method="linear")
            plt.figure(figsize=(18, 10))
            im = plt.contourf(
                grid_lon,
                grid_depth,
                grid_values_lon,
                levels=np.linspace(0, 100, 101),
                cmap=custom_cmap,
                norm=norm,
                extend="both",
            )
            plt.gca().invert_yaxis()
            plt.xlabel("Longitude", fontsize=40)
            plt.plot(
                epicenter_lon,
                hypocenter_depth,
                marker="*",
                color="red",
                markersize=20,
                markeredgecolor="black",
            )
            cbar = plt.colorbar(im)
            cbar.ax.set_title(r"$\mathrm{P_{un}}$ (%)", fontsize=20, pad=15)
            plt.tight_layout()
            plt.savefig(output_dir_lon / f"depth_vs_lon_day{day - 365:04d}.png")
            plt.close()

        points_lat, values_lat = [], []
        for depth in depths:
            for lat in lats:
                vals = [
                    data_dict[(depth, lat, lon)][day]
                    for lon in lons
                    if (depth, lat, lon) in data_dict
                    and not np.isnan(data_dict[(depth, lat, lon)][day])
                ]
                if vals:
                    points_lat.append((lat, depth))
                    values_lat.append(float(np.mean(vals)))

        if len(points_lat) >= 3:
            grid_lat, grid_depth = np.meshgrid(
                np.linspace(min(lats), max(lats), 100),
                np.linspace(min(depths), max(depths), 200),
            )
            grid_values_lat = griddata(points_lat, values_lat, (grid_lat, grid_depth), method="linear")
            plt.figure(figsize=(18, 10))
            im = plt.contourf(
                grid_lat,
                grid_depth,
                grid_values_lat,
                levels=np.linspace(0, 100, 101),
                cmap=custom_cmap,
                norm=norm,
                extend="both",
            )
            plt.gca().invert_yaxis()
            plt.xlabel("Latitude", fontsize=40)
            plt.plot(
                epicenter_lat,
                hypocenter_depth,
                marker="*",
                color="red",
                markersize=20,
                markeredgecolor="black",
            )
            cbar = plt.colorbar(im)
            cbar.ax.set_title(r"$\mathrm{P_{un}}$ (%)", fontsize=20, pad=15)
            plt.tight_layout()
            plt.savefig(output_dir_lat / f"depth_vs_lat_day{day - 365:04d}.png")
            plt.close()

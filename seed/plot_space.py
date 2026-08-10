"""Spatial heatmap plotting from per-grid-point probability CSVs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata


def load_grid_statistics(data_dir: str | Path) -> dict[tuple[float, float], pd.DataFrame]:
    """Load statistics_lat_*_lon_*.csv files into {(lat, lon): DataFrame}."""
    data_dir = Path(data_dir)
    grid_data = {}
    pattern = str(data_dir / "statistics_lat_*.csv")
    for file in glob(pattern):
        filename = Path(file).name
        match = re.match(r"statistics_lat_([-\d.]+)_lon_([-\d.]+)\.csv", filename)
        if not match:
            continue
        lat = float(match.group(1))
        lon = float(match.group(2))
        df = pd.read_csv(file)
        # Normalize column names
        if "Avg_Probability" not in df.columns and "Mean_EQ_Probability(%)" in df.columns:
            df = df.rename(columns={"Mean_EQ_Probability(%)": "Avg_Probability"})
        if "Window_Index" in df.columns and "Avg_Probability" in df.columns:
            grid_data[(lat, lon)] = df
    return grid_data


def plot_space_heatmaps(
    data_dir: str | Path,
    output_dir: str | Path,
    lat_min: float = 31.0,
    lat_max: float = 35.0,
    lon_min: float = 101.5,
    lon_max: float = 105.5,
    hypocenter: tuple[float, float] = (103.82, 33.2),
    fault_shapefile: str | Path | None = None,
    event_day_index: int = 365,
    max_days: int | None = None,
) -> Path:
    """
    Interpolate daily probability fields and save heatmaps.

    fault_shapefile is optional; when omitted, only the probability field is drawn.
    Cartopy/geopandas are imported only when a shapefile is provided or when
    cartopy is available for basemap features.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_data = load_grid_statistics(data_dir)
    if not grid_data:
        raise FileNotFoundError(f"No statistics_lat_*_lon_*.csv files found in {data_dir}")

    example_key = next(iter(grid_data))
    days = grid_data[example_key]["Window_Index"].values
    if max_days is not None:
        days = days[:max_days]

    gdf_faults_subset = None
    if fault_shapefile is not None:
        import geopandas as gpd

        gdf_faults = gpd.read_file(fault_shapefile)
        gdf_faults_subset = gdf_faults.cx[lon_min:lon_max, lat_min:lat_max]

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        use_cartopy = True
    except ImportError:
        ccrs = None
        cfeature = None
        use_cartopy = False

    colors = ["#b3e5fc", "#fff176", "#ff7043", "#d32f2f"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    colorbar_levels = np.linspace(0, 100, 51)

    for idx, k in enumerate(days):
        data_points = []
        values = []
        for (lat, lon), df in grid_data.items():
            row = df[df["Window_Index"] == k]
            if row.empty:
                continue
            data_points.append([lon, lat])
            values.append(float(row.iloc[0]["Avg_Probability"]))

        if len(data_points) < 4:
            continue

        data_points = np.array(data_points)
        values = np.array(values)
        # Add corner anchors for smoother interpolation near boundaries
        boundary = [
            [lon_min, lat_min],
            [lon_min, lat_max],
            [lon_max, lat_min],
            [lon_max, lat_max],
        ]
        boundary_values = []
        for b_lon, b_lat in boundary:
            distances = np.sqrt(
                (data_points[:, 0] - b_lon) ** 2 + (data_points[:, 1] - b_lat) ** 2
            )
            boundary_values.append(values[int(np.argmin(distances))])
        all_points = np.vstack([data_points, boundary])
        all_values = np.hstack([values, boundary_values])

        grid_lon = np.linspace(lon_min, lon_max, 100)
        grid_lat = np.linspace(lat_min, lat_max, 100)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
        try:
            grid_z = griddata(all_points, all_values, (grid_x, grid_y), method="cubic")
            if np.isnan(grid_z).any():
                grid_z_linear = griddata(all_points, all_values, (grid_x, grid_y), method="linear")
                nan_mask = np.isnan(grid_z)
                grid_z[nan_mask] = grid_z_linear[nan_mask]
                if np.isnan(grid_z).any():
                    grid_z_nearest = griddata(
                        all_points, all_values, (grid_x, grid_y), method="nearest"
                    )
                    nan_mask = np.isnan(grid_z)
                    grid_z[nan_mask] = grid_z_nearest[nan_mask]
        except Exception:
            grid_z = griddata(all_points, all_values, (grid_x, grid_y), method="nearest")
        grid_z = np.clip(grid_z, 0, 100)

        fig = plt.figure(figsize=(18, 10))
        if use_cartopy and ccrs is not None:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            cs = ax.contourf(
                grid_x,
                grid_y,
                grid_z,
                levels=colorbar_levels,
                cmap=custom_cmap,
                transform=ccrs.PlateCarree(),
                extend="both",
                vmin=0,
                vmax=100,
            )
            if gdf_faults_subset is not None:
                for geom in gdf_faults_subset.geometry:
                    if geom is None:
                        continue
                    if geom.geom_type == "LineString":
                        x, y = geom.xy
                        ax.plot(
                            x,
                            y,
                            color="black",
                            linewidth=2,
                            alpha=0.5,
                            transform=ccrs.PlateCarree(),
                            zorder=10,
                        )
                    elif geom.geom_type == "MultiLineString":
                        for line in geom.geoms:
                            x, y = line.xy
                            ax.plot(
                                x,
                                y,
                                color="black",
                                linewidth=2,
                                alpha=0.5,
                                transform=ccrs.PlateCarree(),
                                zorder=10,
                            )
            ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor="black")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.plot(
                hypocenter[0],
                hypocenter[1],
                marker="*",
                color="red",
                markersize=30,
                markeredgecolor="black",
                transform=ccrs.PlateCarree(),
            )
            ax.text(
                hypocenter[0] - 0.3,
                hypocenter[1] - 0.3,
                "hypocenter",
                fontsize=40,
                transform=ccrs.PlateCarree(),
            )
            gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax = plt.axes()
            cs = ax.contourf(
                grid_x, grid_y, grid_z, levels=colorbar_levels, cmap=custom_cmap, extend="both"
            )
            if gdf_faults_subset is not None:
                for geom in gdf_faults_subset.geometry:
                    if geom is None:
                        continue
                    if geom.geom_type == "LineString":
                        x, y = geom.xy
                        ax.plot(x, y, color="black", linewidth=1.5, alpha=0.5, zorder=10)
                    elif geom.geom_type == "MultiLineString":
                        for line in geom.geoms:
                            x, y = line.xy
                            ax.plot(x, y, color="black", linewidth=1.5, alpha=0.5, zorder=10)
            ax.plot(
                hypocenter[0],
                hypocenter[1],
                marker="*",
                color="red",
                markersize=20,
                markeredgecolor="black",
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)

        cbar = plt.colorbar(cs, ax=ax, shrink=0.75)
        cbar.ax.set_title(r"$\mathrm{P_{un}}$ (%)", fontsize=20, pad=15)
        cbar.set_ticks([0, 20, 40, 60, 80, 100])

        save_path = output_dir / f"heatmap_day_{int(k) - event_day_index:03d}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    return output_dir

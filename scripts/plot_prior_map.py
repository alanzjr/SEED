#!/usr/bin/env python3
"""Compute and plot the spatially varying prior (SVP) map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seed.config import DEFAULT_CATALOG_2000_2013, OUTPUTS_DIR, ensure_output_dirs
from seed.prior import compute_svp_grid, plot_svp_map


def parse_args():
    p = argparse.ArgumentParser(description="Plot SVP prior probability map.")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_2000_2013)
    p.add_argument("--start-date", type=str, default="2000-01-01")
    p.add_argument("--end-date", type=str, default="2014-12-31")
    p.add_argument("--lon-min", type=float, default=100.0)
    p.add_argument("--lon-max", type=float, default=105.0)
    p.add_argument("--lat-min", type=float, default=28.0)
    p.add_argument("--lat-max", type=float, default=35.0)
    p.add_argument("--grid-size", type=float, default=0.5)
    p.add_argument("--mag-threshold", type=float, default=5.0)
    p.add_argument("--t-years", type=float, default=10.0)
    p.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "svp_spatial_map.png",
    )
    p.add_argument(
        "--csv-output",
        type=Path,
        default=OUTPUTS_DIR / "svp_grid.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    grid, lon_bins, lat_bins = compute_svp_grid(
        args.catalog,
        start_date=args.start_date,
        end_date=args.end_date,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        grid_size=args.grid_size,
        mag_threshold=args.mag_threshold,
        t_years=args.t_years,
    )
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.csv_output, index=False)
    path = plot_svp_map(grid, lon_bins, lat_bins, args.output, mag_threshold=args.mag_threshold)
    print(f"Saved grid CSV: {args.csv_output}")
    print(f"Saved map: {path}")


if __name__ == "__main__":
    main()

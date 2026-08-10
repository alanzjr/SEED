#!/usr/bin/env python3
"""Plot spatial heatmaps from per-grid probability CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seed.config import (
    DEFAULT_FAULT_SHAPEFILE,
    OUTPUTS_DIR,
    RESULTS_SAMPLES_DIR,
    ensure_output_dirs,
)
from seed.plot_space import plot_space_heatmaps


def parse_args():
    p = argparse.ArgumentParser(description="Plot spatial probability heatmaps.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=RESULTS_SAMPLES_DIR,
        help="Directory with statistics_lat_*_lon_*.csv files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "heatmaps",
    )
    p.add_argument("--lat-min", type=float, default=31.0)
    p.add_argument("--lat-max", type=float, default=35.0)
    p.add_argument("--lon-min", type=float, default=101.5)
    p.add_argument("--lon-max", type=float, default=105.5)
    p.add_argument("--hypocenter-lon", type=float, default=103.82)
    p.add_argument("--hypocenter-lat", type=float, default=33.2)
    p.add_argument(
        "--fault-shapefile",
        type=Path,
        default=DEFAULT_FAULT_SHAPEFILE,
        help=(
            "Fault polyline shapefile (default: data/faults/active_faults_study_region.shp). "
            "Pass an empty string to disable the overlay."
        ),
    )
    p.add_argument(
        "--no-faults",
        action="store_true",
        help="Disable fault overlay even if the default shapefile is present",
    )
    p.add_argument("--event-day-index", type=int, default=365)
    p.add_argument(
        "--max-days",
        type=int,
        default=5,
        help="Limit number of days to plot (default 5 for quick demo; set 0 for all)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    max_days = None if args.max_days == 0 else args.max_days
    fault_path = None if args.no_faults else args.fault_shapefile
    if fault_path is not None and str(fault_path).strip() == "":
        fault_path = None
    if fault_path is not None and not Path(fault_path).exists():
        print(f"Warning: fault shapefile not found ({fault_path}); plotting without faults")
        fault_path = None
    out = plot_space_heatmaps(
        args.data_dir,
        args.output_dir,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        hypocenter=(args.hypocenter_lon, args.hypocenter_lat),
        fault_shapefile=fault_path,
        event_day_index=args.event_day_index,
        max_days=max_days,
    )
    print(f"Heatmaps written under {out}")


if __name__ == "__main__":
    main()

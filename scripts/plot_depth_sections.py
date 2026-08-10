#!/usr/bin/env python3
"""Plot depth-section probability figures from depth-grid CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seed.config import OUTPUTS_DIR, ensure_output_dirs
from seed.plot_depth import plot_depth_sections


def parse_args():
    p = argparse.ArgumentParser(description="Plot depth vs lon/lat probability sections.")
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with statistics_depth_*_lat_*_lon_*.csv files",
    )
    p.add_argument(
        "--output-dir-lon",
        type=Path,
        default=OUTPUTS_DIR / "depth_vs_lon",
    )
    p.add_argument(
        "--output-dir-lat",
        type=Path,
        default=OUTPUTS_DIR / "depth_vs_lat",
    )
    p.add_argument("--lat-min", type=float, default=31.0)
    p.add_argument("--lat-max", type=float, default=35.0)
    p.add_argument("--lon-min", type=float, default=101.5)
    p.add_argument("--lon-max", type=float, default=105.5)
    p.add_argument("--epicenter-lon", type=float, default=103.82)
    p.add_argument("--epicenter-lat", type=float, default=33.2)
    p.add_argument("--hypocenter-depth", type=float, default=10.0)
    p.add_argument("--num-days", type=int, default=566)
    p.add_argument(
        "--max-days",
        type=int,
        default=3,
        help="Limit days plotted for a quick demo (0 = all)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    max_days = None if args.max_days == 0 else args.max_days
    plot_depth_sections(
        args.data_dir,
        args.output_dir_lon,
        args.output_dir_lat,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        epicenter_lon=args.epicenter_lon,
        epicenter_lat=args.epicenter_lat,
        hypocenter_depth=args.hypocenter_depth,
        num_days=args.num_days,
        max_days=max_days,
    )
    print(f"Wrote lon sections to {args.output_dir_lon}")
    print(f"Wrote lat sections to {args.output_dir_lat}")


if __name__ == "__main__":
    main()

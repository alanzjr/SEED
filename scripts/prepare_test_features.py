#!/usr/bin/env python3
"""Build a test feature CSV for a target epicenter and mainshock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seed.config import DEFAULT_CATALOG_2014_2025, SAMPLES_DIR, ensure_output_dirs
from seed.features import compute_test_features_at_point, load_catalog


def parse_args():
    p = argparse.ArgumentParser(description="Prepare test features at a fixed location.")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_2014_2025)
    p.add_argument("--lat", type=float, required=True, help="Target latitude")
    p.add_argument("--lon", type=float, required=True, help="Target longitude")
    p.add_argument(
        "--mainshock-mag",
        type=float,
        default=7.0,
        help="Magnitude used to locate the reference mainshock in the catalog",
    )
    p.add_argument(
        "--mainshock-time",
        type=str,
        default=None,
        help="Optional mainshock time (YYYY-MM-DD); overrides magnitude lookup",
    )
    p.add_argument("--m1", type=float, default=7.0)
    p.add_argument("--nb", type=int, default=365)
    p.add_argument("--radial-distance", type=float, default=120.0)
    p.add_argument("--window-preeq", type=int, default=365)
    p.add_argument("--min-mag", type=float, default=1.0)
    p.add_argument("--max-mag", type=float, default=6.0)
    p.add_argument("--day-start", type=int, default=-394)
    p.add_argument("--day-end", type=int, default=200)
    p.add_argument(
        "--output",
        type=Path,
        default=SAMPLES_DIR / "test_features.csv",
        help="Output feature CSV path",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(args.catalog)
    earthquakes = catalog.rename(columns={"time": "date", "mag": "magnitude"}).copy()
    earthquakes["days"] = (
        earthquakes["date"] - earthquakes["date"].min()
    ).dt.total_seconds() / 86400.0

    if args.mainshock_time:
        t0 = pd.to_datetime(args.mainshock_time)
        time_m1 = (t0 - earthquakes["date"].min()).total_seconds() / 86400.0
    else:
        matches = earthquakes.loc[earthquakes["magnitude"] >= args.mainshock_mag]
        if matches.empty:
            raise ValueError(f"No events with magnitude >= {args.mainshock_mag}")
        # Prefer events near the requested location
        near = matches.copy()
        near["dist"] = (near["latitude"] - args.lat) ** 2 + (near["longitude"] - args.lon) ** 2
        time_m1 = float(near.sort_values("dist").iloc[0]["days"])

    feats = compute_test_features_at_point(
        earthquakes,
        latitude=args.lat,
        longitude=args.lon,
        time_m1=time_m1,
        M1=args.m1,
        Nb=args.nb,
        radial_distance=args.radial_distance,
        window_preEQ=args.window_preeq,
        min_mag=args.min_mag,
        max_mag=args.max_mag,
        day_start=args.day_start,
        day_end=args.day_end,
    )
    # Keep only model feature columns for inference compatibility
    feature_cols = [
        "std_depthEQ",
        "std_intertime",
        "std_lat",
        "std_lon",
        "std_magnitude",
        "std_energy_release",
    ]
    out = feats[feature_cols]
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()

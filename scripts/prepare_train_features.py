#!/usr/bin/env python3
"""Build training feature CSVs from an earthquake catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seed.config import DEFAULT_CATALOG_2000_2013, PROCESSED_DIR, SEED, ensure_output_dirs
from seed.features import compute_large_eq_features, compute_random_eq_features, load_catalog
from seed.model import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Prepare training observables from a catalog.")
    p.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_2000_2013,
        help="Catalog CSV with columns time,mag,latitude,longitude,depth",
    )
    p.add_argument("--output-dir", type=Path, default=PROCESSED_DIR, help="Output directory")
    p.add_argument("--m1", type=float, default=6.0, help="Large-event magnitude threshold")
    p.add_argument("--nb", type=int, default=365)
    p.add_argument("--nf", type=int, default=10)
    p.add_argument("--radial-distance", type=float, default=120.0)
    p.add_argument("--window-preeq", type=int, default=365)
    p.add_argument("--min-mag", type=float, default=1.0)
    p.add_argument("--max-mag", type=float, default=6.0)
    p.add_argument("--random-nodes", type=int, default=10, help="Number of negative samples")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(args.catalog)
    large_eqs = catalog[catalog["mag"] >= args.m1]
    print(f"Loaded {len(catalog)} events; {len(large_eqs)} with M>={args.m1}")

    large_feats = compute_large_eq_features(
        catalog,
        large_eqs,
        M1=args.m1,
        Nb=args.nb,
        Nf=args.nf,
        radial_distance=args.radial_distance,
        window_preEQ=args.window_preeq,
        min_mag=args.min_mag,
        max_mag=args.max_mag,
    )
    out_large = args.output_dir / "Observables.csv"
    large_feats.to_csv(out_large, index=False)
    print(f"Wrote {out_large} ({len(large_feats)} rows)")

    for number in range(1, args.random_nodes + 1):
        random_feats = compute_random_eq_features(
            catalog,
            large_eqs,
            Nb=args.nb,
            window_preEQ=args.window_preeq,
            radial_distance=args.radial_distance,
            min_mag=args.min_mag,
            max_mag=args.max_mag,
            num_nodes=1,
        )
        out_rand = args.output_dir / f"Observables_random_{number}.csv"
        random_feats.to_csv(out_rand, index=False)
        print(f"Wrote {out_rand} ({len(random_feats)} rows)")


if __name__ == "__main__":
    main()

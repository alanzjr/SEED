#!/usr/bin/env python3
"""Plot probability time series from predict.py output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seed.config import DEFAULT_EVENT, EVENTS, OUTPUTS_DIR, ensure_output_dirs, get_event_config
from seed.plot_time import plot_probability_curve


def parse_args():
    p = argparse.ArgumentParser(description="Plot mean probability curve with min-max band.")
    p.add_argument(
        "--event",
        type=str,
        default=DEFAULT_EVENT,
        choices=sorted(EVENTS.keys()),
        help="Case-study event (sets default input/output paths and post-event days)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Statistics CSV from scripts/predict.py",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    p.add_argument("--post-event-days", type=int, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    event_cfg = get_event_config(args.event)
    event_out = OUTPUTS_DIR / args.event
    input_path = args.input or (event_out / "window_statistics_cs.csv")
    output_path = args.output or (event_out / "probability_curve.png")
    post_event_days = (
        args.post_event_days
        if args.post_event_days is not None
        else int(event_cfg.get("post_event_days", 200))
    )
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run: python scripts/predict.py --event {args.event}"
        )
    path = plot_probability_curve(
        input_path, output_path, post_event_days=post_event_days, show=args.show
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

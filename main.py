#!/usr/bin/env python3
"""
Main entry point for SEED.

Default behaviour reproduces the manuscript's primary temporal result for one event:
ensemble inference on the shipped sample features, plus the probability curve.

Examples
--------
python main.py
python main.py --event wenchuan
python main.py --event all
python main.py --mode prior
python main.py --mode space
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Import after path setup for listing events in --help
sys.path.insert(0, str(ROOT))
from seed.config import DEFAULT_EVENT, EVENTS  # noqa: E402


def run_script(script_name: str, *extra_args: str) -> None:
    """Run a script under scripts/ with the same Python interpreter."""
    script = ROOT / "scripts" / script_name
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")
    cmd = [sys.executable, str(script), *extra_args]
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="SEED main entry: reproduce paper results from the repository root."
    )
    p.add_argument(
        "--mode",
        choices=["demo", "prior", "space", "all"],
        default="demo",
        help=(
            "demo: predict + probability curve (default); "
            "prior: SVP map; "
            "space: sample spatial heatmaps; "
            "all: demo + prior + space"
        ),
    )
    p.add_argument(
        "--event",
        type=str,
        default=DEFAULT_EVENT,
        help=(
            "Case-study event for demo mode: "
            + ", ".join(sorted(EVENTS))
            + ", or 'all' to run every event"
        ),
    )
    p.add_argument(
        "--no-contribution-plot",
        action="store_true",
        help="Skip the feature-contribution figure during inference",
    )
    p.add_argument(
        "--max-days",
        type=int,
        default=5,
        help="Days to plot for spatial heatmaps (0 = all; used in space/all modes)",
    )
    return p.parse_args()


def resolve_events(event_arg: str) -> list[str]:
    key = event_arg.strip().lower()
    if key == "all":
        return sorted(EVENTS.keys())
    if key not in EVENTS:
        known = ", ".join(sorted(EVENTS)) + ", all"
        raise SystemExit(f"Unknown event '{event_arg}'. Choose one of: {known}")
    return [key]


def run_demo(event: str, no_contribution_plot: bool = False) -> None:
    predict_args = ["--event", event]
    if no_contribution_plot:
        predict_args.append("--no-contribution-plot")
    run_script("predict.py", *predict_args)
    run_script("plot_probability.py", "--event", event)
    print(f"\nDemo finished for '{event}'. Check outputs/{event}/ for:")
    print("  - window_statistics_cs.csv")
    print("  - hn_features_with_labels.csv")
    print("  - probability_curve.png")
    if not no_contribution_plot:
        print("  - all_features_contribution_and_derivative.png")


def run_prior() -> None:
    run_script("plot_prior_map.py")


def run_space(max_days: int = 5) -> None:
    run_script("plot_space_heatmaps.py", "--max-days", str(max_days))


def main() -> None:
    args = parse_args()
    if args.mode in ("demo", "all"):
        for event in resolve_events(args.event):
            print("\n" + "=" * 60)
            print(f"Running demo for event: {event}")
            print("=" * 60)
            run_demo(event, no_contribution_plot=args.no_contribution_plot)
    if args.mode in ("prior", "all"):
        run_prior()
    if args.mode in ("space", "all"):
        run_space(max_days=args.max_days)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute cosine similarity among LSTM hidden-state (HN) samples and plot EQ vs NEQ.

Typical paper use (Wenchuan + Jiuzhaigou merged):

    python scripts/predict.py --event wenchuan --no-contribution-plot
    python scripts/predict.py --event jiuzhaigou --no-contribution-plot
    python scripts/plot_hn_cosine.py --events wenchuan jiuzhaigou

Or point to an already merged CSV with a y_true / Pred_Label column:

    python scripts/plot_hn_cosine.py --input path/to/hn_merged.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seed.config import DEFAULT_EVENT, EVENTS, OUTPUTS_DIR, ensure_output_dirs, get_event_config
from seed.similarity import run_hn_cosine_analysis


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot HN-feature cosine similarity heatmap (EQ vs NEQ)."
    )
    p.add_argument(
        "--events",
        nargs="+",
        default=None,
        choices=sorted(EVENTS.keys()),
        help="Event names whose outputs/<event>/hn_features_with_labels.csv will be merged",
    )
    p.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=None,
        help="One or more HN feature CSVs (overrides --events)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: outputs/hn_cosine_<events>.png)",
    )
    p.add_argument("--n-samples", type=int, default=100, help="Samples drawn per class")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Label column name (default: auto-detect y_true / Pred_Label / label)",
    )
    p.add_argument(
        "--no-absolute",
        action="store_true",
        help="Do not take absolute value of cosine similarities",
    )
    p.add_argument("--title", type=str, default="Sample Cosine Similarity (EQ vs NEQ)")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()

    if args.input:
        csv_paths = list(args.input)
        event_tag = "custom"
    else:
        event_names = args.events or [DEFAULT_EVENT, "wenchuan"]
        # Keep a stable unique order while allowing user order
        seen = []
        for name in event_names:
            key = name.lower()
            get_event_config(key)  # validate
            if key not in seen:
                seen.append(key)
        csv_paths = []
        for name in seen:
            path = OUTPUTS_DIR / name / "hn_features_with_labels.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run first:\n"
                    f"  python scripts/predict.py --event {name} --no-contribution-plot"
                )
            csv_paths.append(path)
        event_tag = "_".join(seen)

    output = args.output or (OUTPUTS_DIR / f"hn_cosine_{event_tag}.png")
    print(f"Inputs ({len(csv_paths)}):")
    for path in csv_paths:
        print(f"  - {path}")
    print(f"Samples per class: {args.n_samples}")

    out = run_hn_cosine_analysis(
        csv_paths=csv_paths,
        output_path=output,
        n_samples=args.n_samples,
        seed=args.seed,
        absolute=not args.no_absolute,
        label_column=args.label_column,
        title=args.title,
        show=args.show,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run ensemble LSTM–RF inference and export probability / contribution outputs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seed.config import (
    DEFAULT_EVENT,
    EVENTS,
    FEATURE_NAMES,
    HIDDEN_SIZE,
    INPUT_SIZE,
    OUTPUTS_DIR,
    WINDOW_SIZE,
    ensure_output_dirs,
    get_event_config,
)
from seed.model import LSTMModel, create_sliding_window

MODEL_NAME_RE = re.compile(r"^model_(\d+)_event_(\d+)\.pth$")


def discover_model_pairs(model_dir: Path) -> list[tuple[Path, Path, int, int]]:
    """
    Find (lstm_path, rf_path, model_index, event_index) pairs in model_dir.

    Expected names: model_{i}_event_{j}.pth and RF_model_{i}_event_{j}.joblib
    """
    pairs = []
    for pth in sorted(model_dir.glob("model_*_event_*.pth")):
        match = MODEL_NAME_RE.match(pth.name)
        if not match:
            continue
        model_index = int(match.group(1))
        event_index = int(match.group(2))
        rf_path = model_dir / f"RF_model_{model_index}_event_{event_index}.joblib"
        if rf_path.exists():
            pairs.append((pth, rf_path, model_index, event_index))
        else:
            print(f"Warning: missing RF companion for {pth.name}, skipping")
    return pairs


def parse_args():
    p = argparse.ArgumentParser(description="Predict with pretrained SEED models.")
    p.add_argument(
        "--event",
        type=str,
        default=DEFAULT_EVENT,
        choices=sorted(EVENTS.keys()),
        help="Case-study event name (sets default input / model-dir / output-dir)",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Feature CSV (default: data/samples/test_<event>.csv)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory with model_*_event_*.pth / RF_*.joblib (default: models/<event>/)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/<event>/)",
    )
    p.add_argument(
        "--model-index",
        type=int,
        default=None,
        help="Optional: restrict to a single model_* index (otherwise use all pairs found)",
    )
    p.add_argument("--event-start", type=int, default=None, help="Optional lower event fold")
    p.add_argument("--event-end", type=int, default=None, help="Optional upper event fold")
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    p.add_argument("--no-contribution-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs()
    event_cfg = get_event_config(args.event)

    input_path = args.input or event_cfg["test_features"]
    model_dir = args.model_dir or event_cfg["model_dir"]
    output_dir = args.output_dir or (OUTPUTS_DIR / args.event)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Event: {event_cfg['display_name']} ({args.event})")
    print(f"Input: {input_path}")
    print(f"Models: {model_dir}")
    print(f"Output: {output_dir}")

    main_data = pd.read_csv(input_path)
    main_data = main_data.drop(
        columns=["day", "days_to_EQ", "date_large_EQ", "num_EQ", "days_since"],
        errors="ignore",
    )
    missing = [c for c in FEATURE_NAMES if c not in main_data.columns]
    if missing:
        raise KeyError(f"Input is missing required feature columns: {missing}")

    values = main_data[FEATURE_NAMES].values
    x = create_sliding_window(values, args.window_size, with_labels=False)
    x_test = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    pairs = discover_model_pairs(model_dir)
    if args.model_index is not None:
        pairs = [p for p in pairs if p[2] == args.model_index]
    if args.event_start is not None:
        pairs = [p for p in pairs if p[3] >= args.event_start]
    if args.event_end is not None:
        pairs = [p for p in pairs if p[3] <= args.event_end]

    if not pairs:
        raise RuntimeError(
            f"No model pairs found in {model_dir}. "
            "Expected files named model_{i}_event_{j}.pth and RF_model_{i}_event_{j}.joblib"
        )
    print(f"Using {len(pairs)} model pair(s)")

    all_probs = []
    all_feature_contributions = []
    all_hn_features = []

    for lstm_path, rf_path, model_index, event_index in pairs:
        print(f"  loading model_{model_index}_event_{event_index}")
        lstm_model = LSTMModel(INPUT_SIZE, args.hidden_size).eval()
        state = torch.load(lstm_path, map_location="cpu")
        lstm_model.load_state_dict(state)

        if x_test.grad is not None:
            x_test.grad.zero_()
        y_pred, lstm_features = lstm_model(x_test, return_features=True)
        y_pred.sum().backward()

        gradients = x_test.grad.detach().numpy()
        feature_contribution = np.abs(gradients)
        all_feature_contributions.append(feature_contribution.mean(axis=1))

        rf_model = joblib.load(rf_path)
        probs = rf_model.predict_proba(lstm_features.detach().numpy())[:, 1]
        all_probs.append(probs)

        with torch.no_grad():
            _, feats = lstm_model(x_test, return_features=True)
            all_hn_features.append(feats.numpy())

        x_test = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    all_probs_np = np.array(all_probs)
    max_probs = np.max(all_probs_np, axis=0)
    min_probs = np.min(all_probs_np, axis=0)
    mean_probs = np.mean(all_probs_np, axis=0)
    pred_labels = (mean_probs > 0.5).astype(int)

    hn_mean = np.array(all_hn_features).mean(axis=0)
    hn_df = pd.DataFrame(hn_mean, columns=[f"hn_{i}" for i in range(args.hidden_size)])
    hn_df.insert(0, "Window_Index", np.arange(len(mean_probs)))
    hn_df["Mean_Probability"] = mean_probs
    hn_df["Pred_Label"] = pred_labels
    hn_out = output_dir / "hn_features_with_labels.csv"
    hn_df.to_csv(hn_out, index=False)
    print(f"Saved {hn_out}")

    statistics_df = pd.DataFrame(
        {
            "Window_Index": np.arange(len(max_probs)),
            "Max_EQ_Probability(%)": max_probs * 100,
            "Min_EQ_Probability(%)": min_probs * 100,
            "Mean_EQ_Probability(%)": mean_probs * 100,
        }
    )
    stats_out = output_dir / "window_statistics_cs.csv"
    statistics_df.to_csv(stats_out, index=False)
    print(f"Saved {stats_out}")

    if args.no_contribution_plot:
        return

    mean_feature_contribution = np.array(all_feature_contributions).mean(axis=0)
    sample_total = mean_feature_contribution.sum(axis=1, keepdims=True)
    sample_ratio = mean_feature_contribution / (sample_total + 1e-8) * 100
    num_features = len(FEATURE_NAMES)
    y_max_contrib = np.ceil(np.max(sample_ratio) + 5)

    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(18, 18), sharex=True)
    if num_features == 1:
        axes = [axes]

    for idx, feature in enumerate(FEATURE_NAMES):
        ax1 = axes[idx]
        ax1.bar(
            np.arange(len(sample_ratio)),
            sample_ratio[:, idx],
            width=0.5,
            color="tab:blue",
            alpha=0.4,
            label="Contribution",
        )
        ax1.set_ylim(0, y_max_contrib)
        ax1.set_ylabel("Contrib. (%)", fontsize=20, color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=16)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        deriv = np.gradient(sample_ratio[:, idx])
        ax2.plot(np.arange(len(deriv)), deriv, color="black", linewidth=2, label="Derivative")
        ax2.set_ylabel("Delta (%)", fontsize=20, color="black")
        ax2.tick_params(axis="y", labelcolor="black", labelsize=16)
        ax1.set_title(feature, fontsize=22)

    axes[-1].set_xlim(0, len(sample_ratio) - 1)
    axes[-1].set_xlabel("Sample Index", fontsize=20)
    fig.align_ylabels(axes)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.4)
    contrib_path = output_dir / "all_features_contribution_and_derivative.png"
    plt.savefig(contrib_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {contrib_path}")


if __name__ == "__main__":
    main()

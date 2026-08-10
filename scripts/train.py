#!/usr/bin/env python3
"""Train LSTM + Random Forest models with leave-one-segment-out validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seed.config import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_EPOCHS,
    P0_DICT,
    PROCESSED_DIR,
    SEED,
    WINDOW_SIZE,
    ensure_output_dirs,
)
from seed.loss import SVP_BalancedFocalLoss
from seed.model import LSTMModel, create_sliding_window, set_seed, split_into_subsets


def parse_args():
    p = argparse.ArgumentParser(description="Train SEED LSTM–RF models.")
    p.add_argument("--train-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument(
        "--observables",
        type=Path,
        default=None,
        help="Positive observables CSV (default: train-dir/Observables.csv)",
    )
    p.add_argument("--model-number", type=int, default=5)
    p.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data1_path = args.observables or (args.train_dir / "Observables.csv")
    data2_path = args.train_dir / f"Observables_random_{args.model_number}.csv"
    if not data1_path.exists() or data1_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty positives file: {data1_path}")
    if not data2_path.exists() or data2_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty negatives file: {data2_path}")

    use_cols = [
        "std_depthEQ",
        "std_intertime",
        "std_lat",
        "std_lon",
        "std_magnitude",
        "std_energy_release",
        "binary_variable",
        "p0_label",
    ]
    drop_cols = ["day", "days_to_EQ", "date_large_EQ", "num_EQ", "energy_release", "days_since"]

    data1 = pd.read_csv(data1_path)
    data2 = pd.read_csv(data2_path)
    data1.drop(columns=drop_cols, errors="ignore", inplace=True)
    data2.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Allow training CSVs without p0_label by defaulting to key 5
    if "p0_label" not in data1.columns:
        data1["p0_label"] = 5
    if "p0_label" not in data2.columns:
        data2["p0_label"] = 5

    data1 = data1[use_cols].values
    data2 = data2[use_cols].values
    segments = split_into_subsets(data1, 365)

    for leave_idx in range(len(segments)):
        print(f"\n-- Leave-one-out validation: segment {leave_idx + 1}/{len(segments)}")
        val_segment = segments[leave_idx]
        x_val, y_val, p0_val_label = create_sliding_window(val_segment, args.window_size)

        x_train_list, y_train_list, p0_train_list = [], [], []
        for i, segment in enumerate(segments):
            if i == leave_idx:
                continue
            x_seg, y_seg, p0_seg = create_sliding_window(segment, args.window_size)
            x_train_list.append(x_seg)
            y_train_list.append(y_seg)
            p0_train_list.append(p0_seg)

        if data2.shape[0] >= args.window_size:
            x2, y2, p02 = create_sliding_window(data2, args.window_size)
            x_train_list.append(x2)
            y_train_list.append(y2)
            p0_train_list.append(p02)

        x_train = np.vstack(x_train_list)
        y_train = np.concatenate(y_train_list)
        p0_train_label = np.concatenate(p0_train_list)

        p0_train = np.zeros_like(p0_train_label, dtype=np.float32)
        p0_val = np.zeros_like(p0_val_label, dtype=np.float32)
        for k, v in P0_DICT.items():
            p0_train[p0_train_label == k] = v
            p0_val[p0_val_label == k] = v

        x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        p0_train_t = torch.tensor(p0_train, dtype=torch.float32).to(device)
        x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        p0_val_t = torch.tensor(p0_val, dtype=torch.float32).to(device)

        model = LSTMModel(input_size=INPUT_SIZE, hidden_size=args.hidden_size).to(device)
        criterion = SVP_BalancedFocalLoss(alpha=0.25, gamma=2.0, p0_dict=P0_DICT)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = -1

        for epoch in range(args.epochs):
            model.train()
            outputs, _, _ = model(x_train_t)
            loss = criterion(outputs, y_train_t, p0_train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs, _, _ = model(x_val_t)
                val_loss_value = criterion(val_outputs, y_val_t, p0_val_t).item()
                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}: Train Loss={loss.item():.4f}, Val Loss={val_loss_value:.4f}"
                )

        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            _, train_features, _ = model(x_train_t)
            _, val_features, _ = model(x_val_t)

        train_features = train_features.cpu().numpy()
        val_features = val_features.cpu().numpy()
        y_train_np = y_train_t.cpu().numpy().ravel()
        y_val_np = y_val_t.cpu().numpy().ravel()

        rf = RandomForestClassifier(n_estimators=500, random_state=args.seed)
        rf.fit(train_features, y_train_np)
        val_preds_rf = rf.predict(val_features)
        val_probs_rf = rf.predict_proba(val_features)[:, 1]
        acc = accuracy_score(y_val_np, val_preds_rf)
        try:
            auc = roc_auc_score(y_val_np, val_probs_rf)
        except ValueError:
            auc = float("nan")

        model_save_path = args.output_dir / f"model_{args.model_number}_event_{leave_idx + 1}.pth"
        rf_save_path = args.output_dir / f"RF_model_{args.model_number}_event_{leave_idx + 1}.joblib"
        torch.save(best_model_state, model_save_path)
        joblib.dump(rf, rf_save_path)
        print(
            f"Segment {leave_idx + 1}: RF Acc={acc:.4f}, AUC={auc:.4f} "
            f"(best LSTM epoch {best_epoch})"
        )
        print(f"Saved {model_save_path}")
        print(f"Saved {rf_save_path}")

    print("All models finished.")


if __name__ == "__main__":
    main()

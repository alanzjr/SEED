"""LSTM model and sliding-window utilities shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Single-layer LSTM with a sigmoid output head."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_features: bool = False):
        out, (hn, _cn) = self.lstm(x)
        features = self.dropout(hn[-1])
        logits = self.fc(features)
        probs = self.sigmoid(logits)
        if return_features:
            return probs, features
        return probs, features, logits


def create_sliding_window(data, window_size: int, with_labels: bool = True):
    """
    Build sliding windows from a 2-D array or DataFrame.

    When with_labels is True, the last two columns are treated as
    binary_variable and p0_label, matching the training feature layout.
    When with_labels is False, all columns are used as features (inference).
    """
    data_array = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)

    if with_labels:
        features, labels, p0_label = [], [], []
        for i in range(len(data_array) - window_size + 1):
            window = data_array[i : i + window_size]
            features.append(window[:, :-2])
            labels.append(window[-1, -2])
            p0_label.append(window[-1, -1])
        return np.array(features), np.array(labels), np.array(p0_label)

    features = []
    for i in range(len(data_array) - window_size + 1):
        window = data_array[i : i + window_size]
        features.append(window[:, :])
    return np.array(features)


def split_into_subsets(data, subset_size: int = 365):
    """Split a sequence into contiguous blocks of subset_size rows."""
    return [data[i : i + subset_size] for i in range(0, len(data), subset_size)]


def set_seed(seed: int) -> None:
    """Fix random seeds for NumPy, Python, and PyTorch."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

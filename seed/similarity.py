"""Cosine-similarity analysis for LSTM hidden-state (HN) features."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

META_COLUMNS = {
    "Window_Index",
    "Mean_Probability",
    "Pred_Label",
    "y_true",
    "y_pred",
    "label",
}


def load_hn_feature_table(
    csv_path: str | Path,
    label_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load HN features and binary labels from a CSV.

    Label column resolution order when label_column is None:
    y_true -> Pred_Label -> label.
    Feature columns are all remaining numeric columns (typically hn_*).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if label_column is None:
        for candidate in ("y_true", "Pred_Label", "label"):
            if candidate in df.columns:
                label_column = candidate
                break
    if label_column is None or label_column not in df.columns:
        raise KeyError(
            f"{csv_path} must contain a label column "
            f"(y_true / Pred_Label / label). Columns: {list(df.columns[:12])}..."
        )

    feature_cols = [
        c
        for c in df.columns
        if c not in META_COLUMNS and c != label_column and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found in {csv_path}")

    features = df[feature_cols].to_numpy(dtype=np.float64)
    labels = df[label_column].to_numpy()
    # Map string labels if needed
    if labels.dtype.kind in {"U", "O", "S"}:
        mapped = np.array(
            [0 if str(v).upper() in {"0", "NEQ", "NEG", "FALSE"} else 1 for v in labels]
        )
        labels = mapped
    else:
        labels = labels.astype(int)
    return features, labels, feature_cols


def merge_hn_tables(
    csv_paths: list[str | Path],
    label_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate HN feature tables (e.g. Wenchuan + Jiuzhaigou)."""
    feats, labs = [], []
    for path in csv_paths:
        f, y, _ = load_hn_feature_table(path, label_column=label_column)
        feats.append(f)
        labs.append(y)
    return np.vstack(feats), np.concatenate(labs)


def balance_by_label(
    features: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the same number of samples from label 0 and label 1."""
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]
    if len(idx_0) < n_samples or len(idx_1) < n_samples:
        raise ValueError(
            f"Need at least {n_samples} samples per class; "
            f"got NEQ={len(idx_0)}, EQ={len(idx_1)}"
        )
    rng = np.random.default_rng(seed)
    sample_idx_0 = rng.choice(idx_0, n_samples, replace=False)
    sample_idx_1 = rng.choice(idx_1, n_samples, replace=False)
    sample_idx = np.concatenate([sample_idx_0, sample_idx_1])
    return features[sample_idx], labels[sample_idx]


def cosine_similarity_matrix(
    features: np.ndarray,
    absolute: bool = True,
) -> np.ndarray:
    """Pairwise cosine similarity; optionally take absolute values."""
    matrix = cosine_similarity(features)
    if absolute:
        matrix = np.abs(matrix)
    return matrix


def sort_by_label(
    matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder rows/columns so NEQ samples come before EQ samples."""
    order = np.argsort(labels)
    return matrix[order][:, order], labels[order]


def plot_eq_neq_heatmap(
    corr_matrix: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str = "Sample Cosine Similarity (EQ vs NEQ)",
    show: bool = False,
) -> Path:
    """
    Plot a heatmap with samples ordered as NEQ then EQ, plus class separators.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_corr, sorted_labels = sort_by_label(corr_matrix, labels)
    n_0 = int(np.sum(sorted_labels == 0))

    plt.figure(figsize=(18, 15))
    sns.heatmap(
        sorted_corr,
        cmap="coolwarm",
        center=0.5,
        cbar_kws={"shrink": 0.8},
    )
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=25)

    plt.title(title, fontsize=40)
    plt.xlabel("Sample", fontsize=40)
    plt.ylabel("Sample", fontsize=40)

    plt.xticks(
        [n_0 / 2, n_0 + (len(sorted_labels) - n_0) / 2],
        ["NEQ", "EQ"],
        fontsize=35,
        rotation=0,
    )
    plt.yticks(
        [n_0 / 2, n_0 + (len(sorted_labels) - n_0) / 2],
        ["NEQ", "EQ"],
        fontsize=35,
    )
    plt.axvline(n_0, color="black", linestyle="--", linewidth=5)
    plt.axhline(n_0, color="black", linestyle="--", linewidth=5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    return output_path


def run_hn_cosine_analysis(
    csv_paths: list[str | Path],
    output_path: str | Path,
    n_samples: int = 100,
    seed: int = 42,
    absolute: bool = True,
    label_column: str | None = None,
    title: str = "Sample Cosine Similarity (EQ vs NEQ)",
    show: bool = False,
) -> Path:
    """End-to-end: load/merge -> balance -> cosine -> heatmap."""
    if len(csv_paths) == 1:
        features, labels, _ = load_hn_feature_table(csv_paths[0], label_column=label_column)
    else:
        features, labels = merge_hn_tables(csv_paths, label_column=label_column)

    balanced_features, balanced_labels = balance_by_label(
        features, labels, n_samples=n_samples, seed=seed
    )
    corr = cosine_similarity_matrix(balanced_features, absolute=absolute)
    return plot_eq_neq_heatmap(
        corr,
        balanced_labels,
        output_path=output_path,
        title=title,
        show=show,
    )

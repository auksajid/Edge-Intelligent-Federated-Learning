"""
preprocessing.py — Edge-feasible preprocessing pipeline (paper Section IV).

Implements, in order:
  1. sliding_windows()               — window segmentation with configurable stride
  2. fit_normalizer / apply_normalizer — per-feature z-score normalization,
     fit once (e.g. at the ground station on early normal-only data) and
     broadcast to all clients before round 1, so no raw data is shared.
  3. extract_statistical_descriptors() — optional lightweight feature-engineering
     path (mean/std/min/max/diff-std per feature) used by the NumPy surrogate
     model in the paper's reproducible ablation study (Section VI). The full
     CNN-LSTM in model.py consumes normalized raw windows directly and does
     NOT require this step — it is provided for parity with the paper's
     surrogate and for any deployment wanting a non-deep-learning fallback.
"""
from __future__ import annotations
import numpy as np


def sliding_windows(series: np.ndarray, window: int = 50, stride: int = 25):
    """series: (T, d) -> windows: (n_windows, window, d)."""
    T, d = series.shape
    starts = range(0, T - window + 1, stride)
    windows = np.stack([series[s:s + window] for s in starts], axis=0)
    return windows.astype(np.float32)


def fit_normalizer(X: np.ndarray):
    """X: (n, w, d) or (n, d) -> per-feature (mu, sigma) computed over all non-feature axes."""
    axes = tuple(range(X.ndim - 1))
    mu = X.mean(axis=axes)
    sigma = X.std(axis=axes) + 1e-6
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_normalizer(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((X - mu) / sigma).astype(np.float32)


def extract_statistical_descriptors(X: np.ndarray) -> np.ndarray:
    """X: (n, w, d) -> (n, 5*d) descriptor vectors [mean, std, min, max, diff_std] per feature.

    This is the preprocessing step used by the lightweight NumPy surrogate
    model referenced in the paper's Section VI (not required by the full
    CNN-LSTM in model.py, which operates on raw normalized windows).
    """
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    diffstd = np.diff(X, axis=1).std(axis=1)
    return np.concatenate([mean, std, mn, mx, diffstd], axis=1).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 1, (500, 28))
    W = sliding_windows(raw, window=50, stride=25)
    print("Windows:", W.shape)
    mu, sigma = fit_normalizer(W)
    Wn = apply_normalizer(W, mu, sigma)
    print("Normalized window mean/std ~ 0/1:", Wn.mean(), Wn.std())
    desc = extract_statistical_descriptors(Wn)
    print("Descriptors:", desc.shape)

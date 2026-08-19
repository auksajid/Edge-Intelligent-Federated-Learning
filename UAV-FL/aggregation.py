"""
aggregation.py — Server-side aggregation strategies (paper Section III-G/H, Eqs. 6-8).

  * FedAvgAggregator     — plain unweighted / data-size-weighted averaging.
  * AdaptiveAggregator   — Eq. 6-7: weights by data size, staleness decay, and link quality.
  * byzantine_filter     — Eq. 8: discard updates whose norm exceeds 3x the round median.
"""
from __future__ import annotations
import numpy as np


def byzantine_filter(deltas: np.ndarray, weights: np.ndarray, factor: float = 3.0):
    """Eq. 8. deltas: (n_clients, n_params). Returns filtered (deltas, weights).
    Falls back to no filtering if it would remove more than half the participants
    (avoids degenerate aggregation when the malicious fraction is implausibly high)."""
    if len(deltas) <= 4:
        return deltas, weights
    norms = np.linalg.norm(deltas, axis=1)
    median = np.median(norms)
    keep = norms < (factor * median + 1e-6)
    if keep.sum() >= max(3, len(deltas) // 2):
        return deltas[keep], weights[keep]
    return deltas, weights


class FedAvgAggregator:
    """Standard (data-size-weighted) FedAvg — Eq. 6 with w_i = |D_i|."""

    def aggregate(self, deltas: np.ndarray, client_sizes: np.ndarray, **_) -> np.ndarray:
        w = client_sizes.astype(np.float64)
        return (w[:, None] * deltas).sum(0) / (w.sum() + 1e-8)


class AdaptiveAggregator:
    """Staleness- and link-quality-aware adaptive aggregation — Eqs. 6-7.

        w_i = |D_i| * exp(-lambda * staleness_i) * link_quality_i
    """

    def __init__(self, lam: float = 0.15):
        self.lam = lam

    def aggregate(self, deltas: np.ndarray, client_sizes: np.ndarray,
                   staleness: np.ndarray, link_quality: np.ndarray) -> np.ndarray:
        w = client_sizes.astype(np.float64) * np.exp(-self.lam * staleness) * link_quality
        return (w[:, None] * deltas).sum(0) / (w.sum() + 1e-8)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    deltas = rng.normal(0, 1, (10, 50)).astype(np.float32)
    deltas[3] = rng.normal(0, 20, 50)  # inject one Byzantine outlier
    sizes = np.full(10, 200)
    weights = sizes.astype(float)
    filt_d, filt_w = byzantine_filter(deltas, weights)
    print(f"Clients before filtering: {len(deltas)}, after: {len(filt_d)}")

    fed = FedAvgAggregator()
    print("FedAvg result norm:", np.linalg.norm(fed.aggregate(deltas, sizes)))
    print("FedAvg (filtered) result norm:", np.linalg.norm(fed.aggregate(filt_d, sizes[:len(filt_d)])))

    ada = AdaptiveAggregator(lam=0.15)
    staleness = np.zeros(10); link_q = np.full(10, 0.9)
    print("Adaptive result norm:", np.linalg.norm(ada.aggregate(deltas, sizes, staleness, link_q)))

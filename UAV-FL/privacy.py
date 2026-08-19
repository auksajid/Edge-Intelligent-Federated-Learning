"""
privacy.py — Differential privacy for federated updates (paper Section III-F, Eq. 5).

    delta_tilde = clip(delta, -C, C) + 1_S * N(0, sigma^2),   sigma = (C/eps) * kappa

Noise is restricted to the already-sparse transmitted support S so that DP
protection does not undo the communication savings from top-k sparsification
(see compression.py) — this is the fix documented in the paper's Section
III-F and Table IX ablation (a naive implementation that densifies the
vector destroys the compression benefit).
"""
from __future__ import annotations
import numpy as np


def clip_and_add_dp_noise(delta: np.ndarray, support_mask: np.ndarray, epsilon: float,
                           clip_norm: float = 1.0, kappa: float = 0.05,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply Eq. 5. `support_mask` is the boolean mask of transmitted (non-zero) coordinates,
    typically `delta != 0` after topk_sparsify(). If epsilon is None, DP is disabled (identity)."""
    if epsilon is None:
        return delta
    rng = rng or np.random.default_rng()
    clipped = np.clip(delta, -clip_norm, clip_norm)
    sigma = (clip_norm / epsilon) * kappa
    noise = rng.normal(0, sigma, size=delta.shape).astype(np.float32)
    return clipped + noise * support_mask


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = np.zeros(1000, dtype=np.float32)
    idx = rng.choice(1000, 100, replace=False)
    d[idx] = rng.normal(0, 0.5, 100)
    mask = d != 0
    for eps in [0.5, 1.0, 5.0, None]:
        out = clip_and_add_dp_noise(d, mask, epsilon=eps, rng=rng)
        nz = np.count_nonzero(out)
        print(f"epsilon={eps}: nonzero after DP = {nz} (sparsity preserved: {nz <= 100 + 5})")

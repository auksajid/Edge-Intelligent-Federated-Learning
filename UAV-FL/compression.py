"""
compression.py — Communication-efficient update compression (paper Section III-E, Eqs. 3-4).

Operates on a flattened 1-D update vector (see federated.flatten_state_dict /
unflatten_state_dict) so it is architecture-agnostic: the same functions
compress an update from the small NumPy surrogate or from the full
CNN-LSTM's ~55K-parameter state_dict.
"""
from __future__ import annotations
import numpy as np


def topk_sparsify(delta: np.ndarray, frac: float) -> np.ndarray:
    """Eq. 3: keep only the top `frac` fraction of coordinates by magnitude, zero the rest."""
    if frac >= 1.0:
        return delta.copy()
    k = max(1, int(len(delta) * frac))
    idx = np.argpartition(np.abs(delta), -k)[-k:]
    out = np.zeros_like(delta)
    out[idx] = delta[idx]
    return out


def quantize_int8(vec: np.ndarray):
    """Eq. 4: symmetric linear 8-bit quantization. Returns (int8_codes, scale)."""
    vmax = float(np.max(np.abs(vec))) + 1e-8
    codes = np.round(vec / vmax * 127).astype(np.int8)
    return codes, vmax


def dequantize_int8(codes: np.ndarray, scale: float) -> np.ndarray:
    return (codes.astype(np.float32) / 127.0) * scale


def payload_bytes(delta: np.ndarray) -> int:
    """Bytes actually transmitted for a (possibly sparse) int8-quantized update:
    1 byte per non-zero value + 1 byte per index (amortized sparse-format cost),
    matching the accounting used in the paper's communication-overhead tables."""
    nz = int(np.count_nonzero(delta))
    return 2 * nz


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1, 55489).astype(np.float32)
    sparse = topk_sparsify(d, 0.10)
    codes, scale = quantize_int8(sparse)
    recon = dequantize_int8(codes, scale)
    print(f"Original nonzero: {np.count_nonzero(d)} | After top-10%: {np.count_nonzero(sparse)}")
    print(f"Payload bytes: {payload_bytes(sparse):,}  (vs. {d.nbytes:,} bytes uncompressed fp32)")
    print(f"Quantization error (max abs): {np.max(np.abs(sparse - recon)):.5f}")

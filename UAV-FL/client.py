"""
client.py — UAV client-side local training and update preparation (Algorithm 2).

Each UAVClient holds its own private data and never exposes it. It only
returns a compressed, quantized, differentially-private update to the
ground station via `local_round()`.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import WindowDataset
from .compression import topk_sparsify, quantize_int8, dequantize_int8, payload_bytes
from .privacy import clip_and_add_dp_noise


def flatten_state_dict(sd: dict) -> tuple[np.ndarray, list]:
    """Flatten a PyTorch state_dict into a single 1-D numpy vector + shape metadata for restoring it."""
    shapes = [(k, tuple(v.shape)) for k, v in sd.items()]
    flat = np.concatenate([v.detach().cpu().numpy().ravel() for v in sd.values()])
    return flat.astype(np.float32), shapes


def unflatten_to_state_dict(flat: np.ndarray, shapes: list, reference_sd: dict) -> dict:
    out = {}
    i = 0
    for (k, shape), ref_tensor in zip(shapes, reference_sd.values()):
        n = int(np.prod(shape))
        arr = flat[i:i + n].reshape(shape)
        out[k] = torch.as_tensor(arr, dtype=ref_tensor.dtype)
        i += n
    return out


class UAVClient:
    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray, link_quality: float,
                 local_epochs: int = 5, batch_size: int = 32, lr: float = 1e-3, device: str = "cpu"):
        self.id = client_id
        self.dataset = WindowDataset(X, y)
        self.n = len(y)
        self.link_quality = link_quality
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.staleness = 0

    def local_round(self, model_template: nn.Module, global_state: dict,
                     compression_frac: float = 0.10, quantize: bool = True,
                     dp_epsilon: float | None = 1.0, dp_clip: float = 1.0,
                     is_byzantine: bool = False, seed: int = 0):
        """Runs Algorithm 2 locally, then compresses/quantizes/DP-protects the resulting
        update per Eqs. 3-5. Returns (compressed_delta, n_samples, payload_bytes, shapes)."""
        torch.manual_seed(seed)
        model = copy.deepcopy(model_template).to(self.device)
        model.load_state_dict(global_state)
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        local_flat, shapes = flatten_state_dict(model.state_dict())
        global_flat, _ = flatten_state_dict(global_state)
        delta = local_flat - global_flat

        if is_byzantine:
            rng = np.random.default_rng(seed + 999)
            delta = rng.normal(0, 5.0, size=delta.shape).astype(np.float32)

        delta = topk_sparsify(delta, compression_frac)
        support = delta != 0
        if quantize:
            codes, scale = quantize_int8(delta)
            delta = dequantize_int8(codes, scale)
        if dp_epsilon is not None:
            delta = clip_and_add_dp_noise(delta, support, epsilon=dp_epsilon, clip_norm=dp_clip,
                                           rng=np.random.default_rng(seed + 1))
        n_bytes = payload_bytes(np.where(support, delta, 0.0))
        self.staleness = 0
        return delta, self.n, n_bytes, shapes

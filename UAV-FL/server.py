"""
server.py — Ground-station orchestration (Algorithm 1, server side).

The GroundStation never sees raw data or full local models — only the
compressed, quantized, (optionally DP-protected) update vectors returned
by UAVClient.local_round(). It applies Byzantine filtering, computes
adaptive aggregation weights, updates the global model, and broadcasts it.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .client import flatten_state_dict, unflatten_to_state_dict
from .aggregation import byzantine_filter, FedAvgAggregator, AdaptiveAggregator


class GroundStation:
    def __init__(self, model: nn.Module, aggregator=None, robust_filter: bool = True,
                 byzantine_factor: float = 3.0):
        self.model = model
        self.aggregator = aggregator or AdaptiveAggregator(lam=0.15)
        self.robust_filter = robust_filter
        self.byzantine_factor = byzantine_factor
        self.staleness = {}   # client_id -> rounds since last contribution

    def global_state(self) -> dict:
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def aggregate_round(self, contributions: list):
        """contributions: list of dicts with keys
           {client_id, delta, n_samples, link_quality} for participating clients this round.
        Applies Eq. 8 (Byzantine filtering) then Eq. 6-7 (adaptive aggregation) and updates
        the global model in place. Also advances staleness bookkeeping for ALL known clients."""
        global_flat, shapes = flatten_state_dict(self.global_state())

        deltas = np.stack([c["delta"] for c in contributions])
        sizes = np.array([c["n_samples"] for c in contributions], dtype=np.float64)
        link_q = np.array([c["link_quality"] for c in contributions], dtype=np.float64)
        stale = np.array([self.staleness.get(c["client_id"], 0) for c in contributions], dtype=np.float64)
        weights_precursor = sizes  # used only for the size term inside aggregator

        if self.robust_filter:
            kept_idx = self._filtered_indices(deltas)
            deltas, sizes, link_q, stale = (deltas[kept_idx], sizes[kept_idx],
                                             link_q[kept_idx], stale[kept_idx])

        if isinstance(self.aggregator, AdaptiveAggregator):
            agg_delta = self.aggregator.aggregate(deltas, sizes, stale, link_q)
        else:
            agg_delta = self.aggregator.aggregate(deltas, sizes)

        new_flat = global_flat + agg_delta
        new_state = unflatten_to_state_dict(new_flat, shapes, self.global_state())
        self.model.load_state_dict(new_state)

        participating_ids = {c["client_id"] for c in contributions}
        for cid in list(self.staleness.keys()) + [c["client_id"] for c in contributions]:
            if cid in participating_ids:
                self.staleness[cid] = 0
            else:
                self.staleness[cid] = self.staleness.get(cid, 0) + 1

    def _filtered_indices(self, deltas: np.ndarray) -> np.ndarray:
        if len(deltas) <= 4:
            return np.arange(len(deltas))
        norms = np.linalg.norm(deltas, axis=1)
        median = np.median(norms)
        keep = norms < (self.byzantine_factor * median + 1e-6)
        if keep.sum() >= max(3, len(deltas) // 2):
            return np.where(keep)[0]
        return np.arange(len(deltas))

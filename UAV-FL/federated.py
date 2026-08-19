"""
federated.py — Full federated training loop (Algorithm 1 of the paper).

`run_federated_training` wires together UAVClient (Algorithm 2 + Eqs. 3-5),
GroundStation (Eqs. 6-8), and periodic held-out evaluation, and is the
single entry point `train.py` calls. It supports every ablation reported
in the paper's Section VII (compression fraction, quantization on/off,
adaptive vs. plain FedAvg, DP epsilon, connectivity dropout, Byzantine
fraction, robust filtering on/off) via keyword arguments.
"""
from __future__ import annotations
import time
import numpy as np
import torch

from .client import UAVClient
from .server import GroundStation
from .aggregation import FedAvgAggregator, AdaptiveAggregator
from .metrics import evaluate


def run_federated_training(
    model_template: torch.nn.Module,
    clients_data: list,                 # list of dicts: {id, X, y, link_quality}
    X_test: np.ndarray, y_test: np.ndarray,
    rounds: int = 60,
    clients_per_round: int = 20,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    compression_frac: float = 0.10,
    quantize: bool = True,
    adaptive_aggregation: bool = True,
    dp_epsilon: float | None = 1.0,
    dp_clip: float = 1.0,
    dropout_rate: float = 0.0,
    byzantine_frac: float = 0.0,
    robust_filter: bool = True,
    device: str = "cpu",
    seed: int = 0,
    eval_every: int = 5,
    verbose: bool = True,
):
    """Returns (trained_model, history) where history is a list of per-eval-round dicts."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    clients = [
        UAVClient(c["id"], c["X"], c["y"], c["link_quality"],
                  local_epochs=local_epochs, batch_size=batch_size, lr=lr, device=device)
        for c in clients_data
    ]
    aggregator = AdaptiveAggregator(lam=0.15) if adaptive_aggregation else FedAvgAggregator()
    gs = GroundStation(model_template, aggregator=aggregator, robust_filter=robust_filter)

    history = []
    t0 = time.time()
    for r in range(1, rounds + 1):
        available = [cl for cl in clients if rng.random() > dropout_rate]
        if not available:
            available = clients
        chosen = list(rng.choice(available, size=min(clients_per_round, len(available)), replace=False))

        global_state = gs.global_state()
        contributions, round_bytes = [], 0
        for cl in chosen:
            is_byz = rng.random() < byzantine_frac
            delta, n_samples, n_bytes, _ = cl.local_round(
                model_template, global_state,
                compression_frac=compression_frac, quantize=quantize,
                dp_epsilon=dp_epsilon, dp_clip=dp_clip,
                is_byzantine=is_byz, seed=int(rng.integers(1e6)),
            )
            contributions.append({"client_id": cl.id, "delta": delta, "n_samples": n_samples,
                                   "link_quality": cl.link_quality})
            round_bytes += n_bytes

        gs.aggregate_round(contributions)

        if r % eval_every == 0 or r == rounds:
            metrics = evaluate(gs.model, X_test, y_test, device=device)
            metrics.pop("probs", None); metrics.pop("pred", None); metrics.pop("confusion_matrix", None)
            metrics["round"] = r
            metrics["round_comm_bytes"] = round_bytes
            metrics["elapsed_s"] = time.time() - t0
            history.append(metrics)
            if verbose:
                print(f"[round {r:3d}/{rounds}] acc={metrics['accuracy']:.4f} "
                      f"f1={metrics['f1']:.4f} comm={round_bytes/1024:.1f} KB "
                      f"elapsed={metrics['elapsed_s']:.1f}s")

    return gs.model, history

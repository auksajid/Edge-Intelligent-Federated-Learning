#!/usr/bin/env python3
"""
train.py — End-to-end federated training entry point.

Quick start (synthetic data, matches the paper's default configuration):
    python3 train.py

Common overrides:
    python3 train.py --rounds 100 --clients_per_round 25 --dp_epsilon 2.0
    python3 train.py --no_dp --no_adaptive --compression_frac 1.0   # plain FedAvg baseline
    python3 train.py --byzantine_frac 0.2 --no_robust_filter        # stress-test without defenses

To train on a real dataset instead of the synthetic benchmark, replace the
`build_synthetic_clients()` call below with your own loader that returns the
same `clients_data` / `X_test, y_test` structure (see uavfl/data.py
`load_public_dataset_stub` for per-dataset notes on ALFA / UAV-SEAD /
UAVCAN / Drone-Anomaly).
"""
import argparse
import json
import os
import numpy as np
import torch

from config import Config
from uavfl.data import SyntheticUAVSwarmDataset, ANOMALY_TYPES
from uavfl.preprocessing import fit_normalizer, apply_normalizer
from uavfl.model import CNNLSTMAnomalyDetector, count_parameters, quantized_size_bytes
from uavfl.federated import run_federated_training
from uavfl.metrics import evaluate, per_anomaly_type_recall


def build_synthetic_clients(cfg: Config):
    """Builds the reproducible synthetic non-IID UAV swarm benchmark (paper Table II/IV),
    fits normalization once (as the ground station would broadcast before round 1), and
    splits each client's data into train/test."""
    ds = SyntheticUAVSwarmDataset(
        n_clients=cfg.n_clients, windows_per_client=cfg.windows_per_client,
        window=cfg.window, n_features=cfg.n_features, non_iid=cfg.non_iid, seed=cfg.data_seed,
    )
    X_all, y_all, atype_all = ds.pooled()
    mu, sigma = fit_normalizer(X_all)

    rng = np.random.default_rng(cfg.data_seed)
    clients_data, test_X, test_y, test_atype = [], [], [], []
    for c in ds.clients:
        Xn = apply_normalizer(c["X"], mu, sigma)
        n = len(c["y"])
        idx = rng.permutation(n)
        n_test = max(1, int(0.2 * n))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        clients_data.append({
            "id": c["id"], "X": Xn[train_idx], "y": c["y"][train_idx].astype(np.float32),
            "link_quality": c["link_quality"],
        })
        test_X.append(Xn[test_idx]); test_y.append(c["y"][test_idx])
        test_atype.append(c["atype"][test_idx])

    X_test = np.concatenate(test_X); y_test = np.concatenate(test_y).astype(np.float32)
    atype_test = np.concatenate(test_atype)
    return clients_data, X_test, y_test, atype_test


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Federated CNN-LSTM UAV anomaly detection training")
    for field, default in cfg.__dict__.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{field}", action="store_true", default=default)
            parser.add_argument(f"--no_{field.replace('adaptive_aggregation','adaptive').replace('robust_filter','robust_filter')}",
                                 dest=field, action="store_false")
        else:
            parser.add_argument(f"--{field}", type=type(default) if default is not None else str, default=default)
    parser.add_argument("--no_dp", action="store_true", help="Disable differential privacy (sets dp_epsilon=None)")
    args = parser.parse_args()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if args.no_dp:
        cfg.dp_epsilon = None

    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 70)
    print("Building synthetic UAV swarm benchmark ...")
    clients_data, X_test, y_test, atype_test = build_synthetic_clients(cfg)
    print(f"  {len(clients_data)} clients | test windows: {len(y_test)} | "
          f"test anomaly rate: {y_test.mean():.3f}")

    model = CNNLSTMAnomalyDetector(
        n_features=cfg.n_features, conv1_filters=cfg.conv1_filters, conv2_filters=cfg.conv2_filters,
        lstm1_units=cfg.lstm1_units, lstm2_units=cfg.lstm2_units, dense_units=cfg.dense_units,
        dropout=cfg.dropout,
    )
    n_params = count_parameters(model)
    print(f"Model: CNN-LSTM, {n_params:,} parameters "
          f"({n_params*4/1024:.1f} KB fp32 / {quantized_size_bytes(model)/1024:.1f} KB int8)")
    print("=" * 70)

    trained_model, history = run_federated_training(
        model_template=model, clients_data=clients_data, X_test=X_test, y_test=y_test,
        rounds=cfg.rounds, clients_per_round=cfg.clients_per_round, local_epochs=cfg.local_epochs,
        batch_size=cfg.batch_size, lr=cfg.lr, compression_frac=cfg.compression_frac,
        quantize=cfg.quantize, adaptive_aggregation=cfg.adaptive_aggregation,
        dp_epsilon=cfg.dp_epsilon, dp_clip=cfg.dp_clip, dropout_rate=cfg.dropout_rate,
        byzantine_frac=cfg.byzantine_frac, robust_filter=cfg.robust_filter,
        device=cfg.device, seed=cfg.seed, eval_every=cfg.eval_every,
    )

    final = evaluate(trained_model, X_test, y_test, device=cfg.device)
    by_type = per_anomaly_type_recall(y_test.astype(int), final["pred"], atype_test, ANOMALY_TYPES)

    print("=" * 70)
    print("FINAL RESULTS")
    print(f"  Accuracy:  {final['accuracy']:.4f}")
    print(f"  Precision: {final['precision']:.4f}")
    print(f"  Recall:    {final['recall']:.4f}")
    print(f"  F1-score:  {final['f1']:.4f}")
    print("  Per-anomaly-type recall:")
    for k, v in by_type.items():
        print(f"    {k:20s} recall={v['recall']:.3f}  (n={v['n_samples']})")

    torch.save(trained_model.state_dict(), os.path.join(cfg.out_dir, "final_model.pt"))
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    with open(os.path.join(cfg.out_dir, "final_metrics.json"), "w") as fh:
        json.dump({"accuracy": final["accuracy"], "precision": final["precision"],
                    "recall": final["recall"], "f1": final["f1"],
                    "confusion_matrix": final["confusion_matrix"],
                    "per_anomaly_type": by_type,
                    "model_params": n_params,
                    "model_size_int8_bytes": quantized_size_bytes(trained_model)},
                   fh, indent=2)
    print(f"\nSaved model + metrics to {cfg.out_dir}/")


if __name__ == "__main__":
    main()

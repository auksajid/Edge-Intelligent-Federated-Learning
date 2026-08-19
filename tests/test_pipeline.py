"""
tests/test_pipeline.py — Minimal sanity checks.

Run with:  python3 -m tests.test_pipeline
(or via pytest / unittest discovery: python3 -m unittest tests.test_pipeline)

These are lightweight integration checks, not a full test suite: they
confirm (1) the model can learn in a plain centralized setting, (2) the
compression/DP pipeline preserves approximate update direction, and (3) a
short federated run executes without error end-to-end.
"""
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from uavfl.data import SyntheticUAVSwarmDataset
from uavfl.preprocessing import fit_normalizer, apply_normalizer
from uavfl.model import CNNLSTMAnomalyDetector
from uavfl.metrics import evaluate
from uavfl.compression import topk_sparsify, quantize_int8, dequantize_int8
from uavfl.federated import run_federated_training


class TestModelLearns(unittest.TestCase):
    def test_centralized_loss_decreases(self):
        torch.manual_seed(0); np.random.seed(0)
        ds = SyntheticUAVSwarmDataset(n_clients=8, windows_per_client=120, seed=0, non_iid="mild")
        X, y, _ = ds.pooled()
        mu, sigma = fit_normalizer(X)
        Xn = apply_normalizer(X, mu, sigma)

        model = CNNLSTMAnomalyDetector(n_features=28)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCELoss()
        loader = DataLoader(
            TensorDataset(torch.as_tensor(Xn, dtype=torch.float32),
                          torch.as_tensor(y.astype(np.float32))),
            batch_size=32, shuffle=True,
        )

        losses = []
        for _ in range(8):
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            losses.append(total / len(y))

        self.assertLess(losses[-1], losses[0], "training loss should decrease over epochs")


class TestCompressionRoundTrip(unittest.TestCase):
    def test_topk_and_quantization_preserve_direction(self):
        rng = np.random.default_rng(0)
        delta = rng.normal(0, 1, 10_000).astype(np.float32)
        sparse = topk_sparsify(delta, frac=0.2)
        self.assertLessEqual(np.count_nonzero(sparse), int(10_000 * 0.2) + 1)
        codes, scale = quantize_int8(sparse)
        recon = dequantize_int8(codes, scale)
        # cosine similarity between original top-20% support and reconstruction should be high
        mask = sparse != 0
        cos = np.dot(sparse[mask], recon[mask]) / (
            np.linalg.norm(sparse[mask]) * np.linalg.norm(recon[mask]) + 1e-8)
        self.assertGreater(cos, 0.99)


class TestFederatedRunsEndToEnd(unittest.TestCase):
    def test_short_federated_run_executes(self):
        torch.manual_seed(0); np.random.seed(0)
        ds = SyntheticUAVSwarmDataset(n_clients=6, windows_per_client=30, seed=0)
        X, y, _ = ds.pooled()
        mu, sigma = fit_normalizer(X)
        clients_data = []
        for c in ds.clients:
            Xn = apply_normalizer(c["X"], mu, sigma)
            clients_data.append({"id": c["id"], "X": Xn, "y": c["y"].astype(np.float32),
                                  "link_quality": c["link_quality"]})
        X_test = apply_normalizer(X[:40], mu, sigma)
        y_test = y[:40].astype(np.float32)

        model = CNNLSTMAnomalyDetector(n_features=28)
        trained_model, history = run_federated_training(
            model_template=model, clients_data=clients_data, X_test=X_test, y_test=y_test,
            rounds=3, clients_per_round=3, local_epochs=1, eval_every=1, verbose=False,
        )
        self.assertEqual(len(history), 3)
        self.assertIn("accuracy", history[-1])


if __name__ == "__main__":
    unittest.main()

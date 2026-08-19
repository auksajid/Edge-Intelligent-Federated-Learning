"""
metrics.py — Evaluation metrics used throughout the paper (Section VI, "Evaluation Metrics").
"""
from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)


@torch.no_grad()
def evaluate(model, X: np.ndarray, y: np.ndarray, device: str = "cpu", batch_size: int = 256):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        p = model(xb).cpu().numpy()
        preds.append(p)
    probs = np.concatenate(preds)
    pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "probs": probs,
        "pred": pred,
    }


def per_anomaly_type_recall(y: np.ndarray, pred: np.ndarray, atype: np.ndarray, anomaly_types):
    out = {}
    for kind in anomaly_types:
        mask = atype == kind
        if mask.sum() == 0:
            continue
        out[kind] = {
            "n_samples": int(mask.sum()),
            "recall": float(recall_score(y[mask], pred[mask], zero_division=0)),
        }
    return out

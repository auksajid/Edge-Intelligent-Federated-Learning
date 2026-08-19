"""
data.py — UAV telemetry data sources.

Provides:
  * SyntheticUAVSwarmDataset — the same reproducible, seeded synthetic
    generator used for the paper's numerical study (non-IID clients, four
    documented anomaly mechanisms). Useful for testing this implementation
    end-to-end without any external data.
  * WindowDataset — a torch.utils.data.Dataset wrapping raw windows.
  * load_public_dataset_stub — documents exactly how to plug in each of the
    four public datasets cited in the paper (Table II). These are NOT
    downloaded automatically (no bundled network access to third-party
    dataset hosts in this environment); each stub raises with instructions
    on obtaining and formatting the real data.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

ANOMALY_TYPES = ["sensor_drift", "gps_spoof_spike", "packet_dropout", "noise_burst"]


def _gen_client_series(n_windows, mean, std, anomaly_rate, window, n_features, rng):
    X = np.zeros((n_windows, window, n_features), dtype=np.float32)
    y = np.zeros(n_windows, dtype=np.int64)
    atype = np.array(["normal"] * n_windows, dtype=object)
    phi = 0.85
    for i in range(n_windows):
        series = np.zeros((window, n_features))
        series[0] = mean + rng.normal(0, std, n_features)
        for t in range(1, window):
            series[t] = mean + phi * (series[t - 1] - mean) + rng.normal(0, std * 0.4, n_features)
        if rng.random() < anomaly_rate:
            kind = rng.choice(ANOMALY_TYPES)
            atype[i] = kind
            n_feat = rng.integers(1, 4)
            feats = rng.choice(n_features, n_feat, replace=False)
            t0 = rng.integers(window // 4, window - 5)
            if kind == "sensor_drift":
                ramp = np.linspace(0, rng.uniform(3, 6) * std[feats].mean(), window - t0)
                series[t0:, feats] += ramp[:, None]
            elif kind == "gps_spoof_spike":
                series[t0:t0 + 3, feats] += rng.uniform(5, 9) * std[feats]
            elif kind == "packet_dropout":
                series[t0:t0 + rng.integers(3, 8), feats] = 0.0
            elif kind == "noise_burst":
                series[t0:, feats] += rng.normal(0, 4 * std[feats], (window - t0, n_feat))
            y[i] = 1
        X[i] = series
    return X, y, atype


class SyntheticUAVSwarmDataset:
    """Reproducible synthetic non-IID UAV swarm telemetry generator.

    Mirrors the benchmark described in Table II / Section V of the paper.
    Use this to validate the federated pipeline end-to-end before switching
    to a real dataset via `load_public_dataset_stub`.
    """

    def __init__(self, n_clients: int = 50, windows_per_client: int = 200, window: int = 50,
                 n_features: int = 28, non_iid: str = "severe", seed: int = 0):
        self.n_clients = n_clients
        self.window = window
        self.n_features = n_features
        rng = np.random.default_rng(seed)
        self.clients = []
        for c in range(n_clients):
            if non_iid == "iid":
                mean = rng.normal(0, 0.3, n_features)
            elif non_iid == "mild":
                mean = rng.normal(0, 1.0, n_features)
            else:
                mean = rng.normal(0, 2.2, n_features)
            std = np.abs(rng.normal(1.0, 0.25, n_features)) + 0.2
            anomaly_rate = float(np.clip(rng.normal(0.10, 0.03), 0.03, 0.20))
            client_rng = np.random.default_rng(seed * 1000 + c + 1)
            X, y, atype = _gen_client_series(windows_per_client, mean, std, anomaly_rate,
                                              window, n_features, client_rng)
            link_quality = float(np.clip(rng.normal(0.85, 0.1), 0.4, 1.0))
            self.clients.append({"id": c, "X": X, "y": y, "atype": atype, "link_quality": link_quality})

    def client_data(self, client_id: int):
        c = self.clients[client_id]
        return c["X"], c["y"], c["atype"], c["link_quality"]

    def pooled(self):
        X = np.concatenate([c["X"] for c in self.clients], axis=0)
        y = np.concatenate([c["y"] for c in self.clients], axis=0)
        atype = np.concatenate([c["atype"] for c in self.clients], axis=0)
        return X, y, atype


class WindowDataset(Dataset):
    """torch Dataset wrapping (X, y) window arrays produced by preprocessing.py."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_public_dataset_stub(name: str):
    """Instructions for plugging in each public dataset cited in the paper (Table II).

    Each of these datasets has a different native format; this function
    intentionally does not attempt to auto-download or parse them (no
    bundled redistribution rights, and formats vary), but documents exactly
    what a `load_<name>()` function should return: (X, y, atype, client_ids)
    with X of shape (n_windows, window, n_features), y in {0,1}, atype a
    string array from ANOMALY_TYPES (or "normal"), and client_ids assigning
    each window to a UAV/flight for federated partitioning.
    """
    instructions = {
        "alfa": (
            "ALFA (Air Lab Failure and Anomaly) — Keipour, Mousaei & Scherer, IJRR 2021.\n"
            "Source: https://github.com/castacks/alfa-dataset\n"
            "47 flight logs (rosbag/csv), engine + 7 actuator/control-surface fault types.\n"
            "Loader sketch: for each flight, parse the timestamped fault-onset label, resample "
            "IMU/GPS/actuator channels to a common rate, apply sliding_windows() from "
            "preprocessing.py, and label windows after the documented fault-onset timestamp as "
            "anomalous (atype='sensor_drift' or map ALFA's specific fault type as appropriate). "
            "Treat each flight (or each aircraft) as one federated client."
        ),
        "uav_sead": (
            "UAV-SEAD (state-estimation anomaly dataset), cited in AeroTSBoost, arXiv:2605.25639.\n"
            "1,389 annotated PX4 flight logs, 87 channels.\n"
            "Loader sketch: parse the PX4 .ulg logs (pyulog), select the 28 channels matching "
            "Table III of the paper (or adapt n_features), align to window=50 @ the dataset's "
            "native rate, and use the provided anomaly annotations directly as labels."
        ),
        "uavcan": (
            "UAVCAN labeled extraction dataset, arXiv:2212.09268.\n"
            "Labeled CAN-bus / MAVLink attack + benign traffic.\n"
            "Loader sketch: parse CAN frames into a fixed-rate multivariate time series (one "
            "column per monitored signal ID), window, and label using the dataset's attack "
            "interval annotations (atype='gps_spoof_spike' for spoofing-type attacks, "
            "'noise_burst' for flooding/DoS-type attacks, mapped per the dataset's attack taxonomy)."
        ),
        "drone_anomaly": (
            "Drone-Anomaly — Jin, Mou, Xia & Zhu; arXiv:2209.13363.\n"
            "Aerial video anomaly dataset (7 scenes, 37 train / 22 test sequences).\n"
            "This dataset is video-domain, not telemetry; using it requires either (a) extracting "
            "a proxy telemetry-like feature vector per frame (e.g., optical-flow statistics) to "
            "reuse this package's CNN-LSTM directly, or (b) treating it as a separate visual "
            "anomaly-detection benchmark evaluated with a vision backbone instead of Table's "
            "telemetry pipeline. Documented here for completeness per Table II of the paper."
        ),
    }
    if name not in instructions:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(instructions)}.")
    raise NotImplementedError(
        f"load_public_dataset_stub('{name}') is a documentation stub, not a working "
        f"downloader (see README.md, 'Moving to Real Datasets').\n\n{instructions[name]}"
    )


if __name__ == "__main__":
    ds = SyntheticUAVSwarmDataset(n_clients=5, windows_per_client=20, seed=1)
    X, y, atype, lq = ds.client_data(0)
    print("Client 0:", X.shape, y.shape, "anomaly rate:", y.mean(), "link quality:", lq)
    Xp, yp, ap = ds.pooled()
    print("Pooled:", Xp.shape, "total anomaly rate:", yp.mean())

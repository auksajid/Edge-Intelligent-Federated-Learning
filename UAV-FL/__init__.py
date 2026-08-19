"""
uavfl — Edge-Intelligent, Communication-Efficient Federated Learning
for Real-Time Anomaly Detection in Resource-Constrained UAV Networks.

Reference PyTorch implementation of the full raw-window CNN-LSTM
architecture and the communication-efficient, Byzantine-robust,
differentially private federated protocol. This is the production-architecture counterpart to
the lightweight NumPy surrogate used for the paper's reproducible ablation
study; see README.md for how the two relate.
"""
from .model import CNNLSTMAnomalyDetector, count_parameters, quantized_size_bytes
from .data import (
    SyntheticUAVSwarmDataset, WindowDataset, ANOMALY_TYPES,
    load_public_dataset_stub,
)
from .preprocessing import (
    sliding_windows, fit_normalizer, apply_normalizer, extract_statistical_descriptors,
)
from .compression import topk_sparsify, quantize_int8, dequantize_int8, payload_bytes
from .privacy import clip_and_add_dp_noise
from .aggregation import FedAvgAggregator, AdaptiveAggregator, byzantine_filter
from .client import UAVClient
from .server import GroundStation
from .federated import run_federated_training
from .metrics import evaluate, per_anomaly_type_recall

__all__ = [
    "CNNLSTMAnomalyDetector", "count_parameters", "quantized_size_bytes",
    "SyntheticUAVSwarmDataset", "WindowDataset", "ANOMALY_TYPES", "load_public_dataset_stub",
    "sliding_windows", "fit_normalizer", "apply_normalizer", "extract_statistical_descriptors",
    "topk_sparsify", "quantize_int8", "dequantize_int8", "payload_bytes",
    "clip_and_add_dp_noise",
    "FedAvgAggregator", "AdaptiveAggregator", "byzantine_filter",
    "UAVClient", "GroundStation", "run_federated_training",
    "evaluate", "per_anomaly_type_recall",
]

__version__ = "1.0.0"

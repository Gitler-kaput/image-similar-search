from __future__ import annotations
import numpy as np

def precision_at_k(relevant: np.ndarray, k: int) -> float:
    relevant = relevant[:k]
    return float(np.mean(relevant)) if len(relevant) else 0.0

def recall_at_k(relevant: np.ndarray, k: int) -> float:
    relevant = relevant[:k]
    return float(np.sum(relevant) > 0)

def average_precision(relevant: np.ndarray, k: int) -> float:
    relevant = relevant[:k]
    if relevant.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, r in enumerate(relevant, start=1):
        if r:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0

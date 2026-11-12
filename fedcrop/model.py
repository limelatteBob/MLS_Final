"""A small CNN without batch-dependent normalization."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class CropCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128), normalize=False):
        super().__init__()
        self.normalize = normalize
        layers, previous = [], 3
        for width in channels:
            layers.extend([nn.Conv2d(previous, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
            previous = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(previous, 10)

    def forward(self, images):
        if self.normalize:
            images = (images - 0.5) * 2
        return self.head(self.features(images).flatten(1))


def get_weights(model):
    return [p.detach().cpu().numpy().copy() for p in model.state_dict().values()]


def set_weights(model, weights):
    state = model.state_dict()
    if len(weights) != len(state):
        raise ValueError("Parameter count mismatch")
    result = {}
    for (name, current), value in zip(state.items(), weights):
        if tuple(current.shape) != tuple(value.shape) or not np.isfinite(value).all():
            raise ValueError(f"Invalid parameter {name}")
        result[name] = torch.as_tensor(value, dtype=current.dtype)
    model.load_state_dict(result)


def classification_metrics(labels, predictions):
    matrix = np.zeros((10, 10), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[int(actual), int(predicted)] += 1
    return metrics_from_confusion(matrix)


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = matrix.diagonal()
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(10), where=matrix.sum(1) != 0)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(10), where=denominator != 0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(tp.sum() / max(1, matrix.sum())),
            "per_class_recall": recall.tolist(), "confusion_matrix": matrix.tolist(),
            "samples": int(matrix.sum())}


def evaluate(model, dataset, device="cpu"):
    model.to(device).eval()
    labels, predictions = [], []
    with torch.inference_mode():
        for images, targets in DataLoader(dataset, batch_size=64):
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            labels.extend(targets.tolist())
    return classification_metrics(labels, predictions)

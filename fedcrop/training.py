"""Local optimization; no private statistics are returned to the coordinator."""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from opacus import GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.optimizers import DPOptimizer

from .model import CropCNN, get_weights, set_weights
from .privacy import BudgetExhausted


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sampling_rate(size, batch_size):
    return 1.0 / math.ceil(size / min(size, batch_size))


def train_local(weights, dataset, config, *, seed, steps, batch_size, ledger=None, sigma=None):
    seed_everything(seed)
    device = config["device"]
    model = CropCNN(normalize=config.get("normalize", False)).to(device)
    set_weights(model, weights)
    if config.get("freeze_features", False):
        model.features.requires_grad_(False)
    model.train()
    batch_size = min(batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer_name = config.get("optimizer", "sgd")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=config["learning_rate"])
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    else:
        raise ValueError("optimizer must be sgd or adam")
    rate = sampling_rate(len(dataset), batch_size)
    if ledger is not None:
        model = GradSampleModule(model)
        loader = DPDataLoader.from_data_loader(loader)
        optimizer = DPOptimizer(optimizer, noise_multiplier=sigma, max_grad_norm=config["clip_norm"],
                                expected_batch_size=max(1, int(len(dataset) * rate)))
    start = time.perf_counter()
    iterator, completed = iter(loader), 0
    for _ in range(steps):
        if ledger is not None:
            try:
                ledger.reserve(sigma, rate)
            except BudgetExhausted:
                break
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        if ledger is not None:
            physical = config["physical_batch_size"]
            parts = max(1, math.ceil(len(images) / physical))
            for part in range(parts):
                optimizer.zero_grad()
                x = images[part * physical:(part + 1) * physical].to(device)
                y = targets[part * physical:(part + 1) * physical].to(device)
                if len(x):
                    torch.nn.functional.cross_entropy(model(x), y).backward()
                else:
                    # Poisson batches may be empty: still release a correctly noised step.
                    for parameter in model.parameters():
                        if parameter.requires_grad:
                            parameter.grad_sample = torch.zeros((0, *parameter.shape), device=device)
                optimizer.signal_skip_step(part < parts - 1)
                optimizer.step()
        else:
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(images.to(device)), targets.to(device)).backward()
            optimizer.step()
        completed += 1
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    base = model._module if ledger is not None else model
    weights = get_weights(base)
    if ledger is not None:
        model.remove_hooks()
    return weights, {"steps": completed, "wall_seconds": time.perf_counter() - start,
                     "sample_rate": rate, "epsilon": ledger.epsilon if ledger else None}

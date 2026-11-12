"""Measured host profiles and explicitly simulated resource constraints."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict

import numpy as np
import psutil
import torch
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer

from .io import atomic_json, host_info
from .model import CropCNN


class PeakRSS:
    def __enter__(self):
        self.peak = psutil.Process().memory_info().rss
        self.stop = threading.Event()

        def sample():
            while not self.stop.wait(0.005):
                self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, psutil.Process().memory_info().rss)


def profile_host(output, repeats=3, freeze_features=False, normalize=False):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    results = []
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        for batch in [8, 16, 32]:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            base = CropCNN(normalize=normalize).to(device)
            if freeze_features:
                base.features.requires_grad_(False)
            model = GradSampleModule(base)
            optimizer = DPOptimizer(torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01),
                                    noise_multiplier=1.0, max_grad_norm=1.0,
                                    expected_batch_size=batch)
            x, y = torch.randn(batch, 3, 96, 96, device=device), torch.arange(batch, device=device) % 10
            timings = []
            with PeakRSS() as memory:
                for i in range(repeats + 1):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    optimizer.zero_grad()
                    torch.nn.functional.cross_entropy(model(x), y).backward()
                    optimizer.step()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    if i:
                        timings.append(time.perf_counter() - start)
            results.append({"device": device, "batch_size": batch,
                            "step_seconds": float(np.median(timings)), "peak_rss_bytes": memory.peak,
                            "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0})
            del model, optimizer, x, y
    result = {"host": host_info(), "training": results, "torch_version": torch.__version__,
              "freeze_features": freeze_features, "normalize": normalize,
              "note": "CPU process RSS includes Python/PyTorch runtime; virtual device times are predictions, energy is a proxy."}
    atomic_json(output, result)
    return result


@dataclass
class Device:
    client_id: int
    profile: str
    ram_mb: float
    compute: float
    uplink_mbps: float
    rtt_seconds: float
    battery: float
    age: int = 0
    participations: int = 0


def make_devices(count=12, seed=2026):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        kind = min(2, i * 3 // count)
        name, ram, scale, bw, rtt = [("camera", 512, 4, 1, 0.15),
                                   ("phone", 2048, 2, 5, 0.06),
                                   ("gateway", 8192, 1, 20, 0.02)][kind]
        battery = float(rng.uniform(0.3, 0.7) if kind == 0 else rng.uniform(0.4, 0.9) if kind == 1 else 1)
        result.append(Device(i, name, ram, scale, bw, rtt, battery))
    return result


def workload(device, profile, config, update_bytes):
    for batch in [b for b in [32, 16, 8] if b <= config["batch_size"]]:
        entry = next(x for x in profile["training"] if x["device"] == "cpu" and x["batch_size"] == batch)
        if entry["peak_rss_bytes"] * config.get("memory_headroom", 1.0) > device.ram_mb * 1024**2:
            continue
        seconds = entry["step_seconds"] * device.compute * config["compute_scale"]
        transfer_bytes = 3 * update_bytes + 30000 if config["secure_aggregation"] else 2 * update_bytes
        stages = 4 if config["secure_aggregation"] else 1
        transfer = transfer_bytes * 8 / (device.uplink_mbps * config["bandwidth_scale"] * 1e6) + stages * device.rtt_seconds
        # Reserve headroom for the configured trace's maximum +10% jitter.
        steps = min(config["local_steps"], int(max(0, config["round_deadline"] / 1.1 - transfer) / max(seconds, 1e-6)))
        if device.profile != "gateway":
            energy_left = device.battery - config["battery_reserve"] - transfer * 0.00004
            steps = min(steps, int(max(0, energy_left) / max(seconds * 0.00002, 1e-12)))
        if steps and (device.profile == "gateway" or device.battery > config["battery_reserve"]):
            return {"batch_size": batch, "steps": steps, "seconds": seconds * steps + transfer,
                    "step_seconds": seconds}
    return None


def select_clients(devices, feasible, count, method, rng):
    candidates = [d for d in devices if d.client_id in feasible]
    if method == "uniform":
        indices = rng.choice(len(candidates), min(count, len(candidates)), replace=False)
        return [candidates[i].client_id for i in indices]
    if method not in {"fair", "speed"}:
        raise ValueError("Unknown scheduling policy")
    reserved = sorted(candidates, key=lambda d: (-d.age, d.client_id))[:min(2, count)] if method == "fair" else []
    rest = sorted((d for d in candidates if d not in reserved), key=lambda d: (feasible[d.client_id]["seconds"], d.client_id))
    return [d.client_id for d in (reserved + rest)[:count]]


def assignment(registry, device, latency_budget_ms=100, compute_scale=1.0):
    possible = [m for m in registry if m["peak_rss_bytes"] <= device.ram_mb * 1024**2
                and m["p95_ms"] * device.compute * compute_scale <= latency_budget_ms]
    if not possible:
        return None
    return min(possible, key=lambda m: (-m["validation"]["macro_f1"], m["artifact_bytes"], m["name"]))["name"]

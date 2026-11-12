"""Demonstrate dynamic versus fixed assignment on measured model artifacts."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedcrop.io import read_json, atomic_json
from fedcrop.deployment import assign_models

parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
args = parser.parse_args()
root = Path(args.run_dir)
initial = assign_models(root)
registry = {m["name"]: m for m in read_json(root / "registry.json")["models"]}
fixed = initial["assignments"]
trajectory = []
for name, latency, memory in [("normal", 100., None), ("tight_latency", 2., None),
                               ("memory_pressure", 100., 1.), ("recovered", 100., None)]:
    devices = [dict(d) for d in initial["devices"]]
    if memory is not None:
        devices[0]["ram_mb"] = memory
    path = root / "resource_trace_devices.json"
    atomic_json(path, devices)
    current = assign_models(root, path, latency)
    violations = 0
    for device in devices:
        model = registry.get(fixed.get(str(device["client_id"])))
        if model and (model["peak_rss_bytes"] > device["ram_mb"] * 1024**2
                      or model["p95_ms"] * device["compute"] > latency):
            violations += 1
    trajectory.append({"scenario": name, "fixed_assignment_budget_violations": violations,
                       "dynamic_paused_clients": sum(v is None for v in current["assignments"].values()),
                       "dynamic_download_bytes": current["download_bytes"], "changes": current["changes"]})
atomic_json(root / "deployment_ablation.json", {"kind": "synthetic_resource_changes", "scenarios": trajectory})
print(trajectory)

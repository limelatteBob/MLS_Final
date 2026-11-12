from dataclasses import asdict
import json

from fedcrop.io import atomic_json, read_json
from fedcrop.resources import make_devices
from fedcrop.deployment import assign_models


def test_assignment_resource_change_and_pause(tmp_path):
    devices = make_devices(3)
    registry = [{"name": "fp32", "artifact_bytes": 1000, "peak_rss_bytes": 100 * 1024**2,
                 "p95_ms": 5., "validation": {"macro_f1": .9}, "quality_acceptable": True},
                {"name": "int8", "artifact_bytes": 300, "peak_rss_bytes": 70 * 1024**2,
                 "p95_ms": 1., "validation": {"macro_f1": .89}, "quality_acceptable": True}]
    atomic_json(tmp_path / "registry.json", {"models": registry})
    atomic_json(tmp_path / "config.json", {"num_clients": 3, "partition_seed": 2026, "compute_scale": 1})
    original = assign_models(tmp_path)
    assert original["assignments"]["0"] == "fp32"
    faster = assign_models(tmp_path, latency_budget_ms=10)
    assert faster["assignments"]["0"] == "int8"
    assert faster["download_bytes"] == 300
    devices[0].ram_mb = 1
    changed = tmp_path / "devices.json"
    atomic_json(changed, [asdict(d) for d in devices])
    paused = assign_models(tmp_path, changed, latency_budget_ms=10)
    assert paused["assignments"]["0"] is None
    assert paused["download_bytes"] == 0


def test_same_variant_new_checkpoint_triggers_download(tmp_path):
    model = {"name": "fp32", "sha256": "v1", "artifact_bytes": 1000, "peak_rss_bytes": 1000,
             "p95_ms": 1., "validation": {"macro_f1": .9}, "quality_acceptable": True}
    atomic_json(tmp_path / "registry.json", {"models": [model]})
    atomic_json(tmp_path / "config.json", {"num_clients": 3, "partition_seed": 2026, "compute_scale": 1})
    assign_models(tmp_path)
    assert assign_models(tmp_path)["download_bytes"] == 0
    model["sha256"] = "v2"
    atomic_json(tmp_path / "registry.json", {"models": [model]})
    assert assign_models(tmp_path)["download_bytes"] == 3000

"""Failure injection across the experiment/checkpoint/privacy boundary."""
from pathlib import Path
import json

import numpy as np
import pytest
from PIL import Image

from fedcrop.io import atomic_json, digest, file_hash, read_json
from fedcrop.experiment import load_config, run
import fedcrop.experiment as experiment


@pytest.mark.parametrize("automatic", [False, True])
def test_crash_then_resume_preserves_spent_budget(tmp_path, monkeypatch, automatic):
    rows = []
    for split, count in [("train", 24), ("validation", 10), ("public", 10), ("test", 10)]:
        for i in range(count):
            path = tmp_path / f"{split}_{i}.png"
            Image.fromarray(np.full((12, 12, 3), (i * 11) % 255, dtype=np.uint8)).save(path)
            rows.append({"path": str(path), "split": split, "label": i % 10,
                         "sha256": file_hash(path), "group_id": f"{split}_{i}"})
    # Distinct file checksums across splits prevent accidental fixture leakage.
    for i, row in enumerate(rows):
        pixels = np.full((12, 12, 3), i, dtype=np.uint8)
        Image.fromarray(pixels).save(row["path"])
        row["sha256"] = file_hash(row["path"])
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, {"rows": rows, "manifest_hash": digest(rows), "all_physical_leaves_verified": True})
    profile = tmp_path / "profile.json"
    atomic_json(profile, {"training": [{"device": "cpu", "batch_size": b, "peak_rss_bytes": 1000,
                                        "step_seconds": .001} for b in [8, 16, 32]]})
    config = load_config("configs/smoke.json", {"output": str(tmp_path / "run"), "manifest": str(manifest),
            "profile_path": str(profile), "device": "cpu", "num_clients": 6, "rounds": 1,
            "local_steps": 1, "noise_multiplier": 10., "export_after_training": False,
            "evaluate_test": False,
            "max_retries": 1 if automatic else 0})
    original_evaluate = experiment.evaluate
    def validation_only(model, dataset, device):
        assert all(r["split"] == "validation" for r in dataset.rows), "Training accessed held-out test data"
        return original_evaluate(model, dataset, device)
    monkeypatch.setattr(experiment, "evaluate", validation_only)
    original = experiment.train_local
    has_crashed = False
    def crash(*args, **kwargs):
        nonlocal has_crashed
        result = original(*args, **kwargs)
        if not has_crashed:
            has_crashed = True
            raise RuntimeError("injected after DP step")
        return result
    monkeypatch.setattr(experiment, "train_local", crash)
    if automatic:
        result = run(config)
    else:
        with pytest.raises(RuntimeError, match="injected"):
            run(config)
    paths = sorted((Path(config["output"]) / "private").glob("*.json"))
    before = {p.name: read_json(p)["reserved_steps"] for p in paths}
    if not automatic:
        assert sum(before.values()) == 1
        monkeypatch.setattr(experiment, "train_local", original)
        result = run(config, resume=True)
    after = {p.name: read_json(p)["reserved_steps"] for p in paths}
    assert sum(after.values()) == 7
    assert all(after[k] >= before[k] for k in before)
    assert result["budget_adherence"] and result["rounds"] == 1
    assert result["test"] is None and result["test_status"] == "withheld"

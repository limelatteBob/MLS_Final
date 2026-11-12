"""Validate actual saved run artifacts without executing training again."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedcrop.io import read_json, file_hash
from fedcrop.data import validate_manifest
from fedcrop.privacy import PrivacyLedger

parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
args = parser.parse_args()
root = Path(args.run_dir)
config = read_json(root / "config.json")
summary = read_json(root / "summary.json")
manifest = read_json(config["manifest"])
validate_manifest(manifest, verify_files=True)
assert summary["manifest_hash"] == manifest["manifest_hash"]
if summary.get("public_checkpoint_sha256"):
    assert file_hash(config["public_checkpoint"]) == summary["public_checkpoint_sha256"]
if (root / "final_test.json").exists():
    final_test = read_json(root / "final_test.json")
    assert final_test["checkpoint_sha256"] == file_hash(root / "best.pt")
    assert final_test["manifest_hash"] == manifest["manifest_hash"]
    assert final_test["config_hash"] == summary["config_hash"]
if config["mode"] == "federated":
    for client in range(config["num_clients"]):
        if config["dp"]:
            ledger = PrivacyLedger(root / "private" / f"client_{client}.json", summary["run_id"], client,
                                   config["epsilon_cap"], config["delta"])
            assert ledger.epsilon <= ledger.cap
    history = read_json(root / "rounds.json")
    version = 0
    for row in history:
        if row["status"] == "committed":
            version += 1
        assert row["model_version"] == version
        assert row["wire_bytes"] >= 0
    if (root / "registry.json").exists():
        registry = read_json(root / "registry.json")
        assert registry["manifest_hash"] == manifest["manifest_hash"]
        for model in registry["models"]:
            assert file_hash(model["path"]) == model["sha256"]
            assert model["p95_ms"] >= model["p50_ms"] > 0
print(f"Validated data checksums, privacy budgets, committed model versions and exported artifacts: {root}")

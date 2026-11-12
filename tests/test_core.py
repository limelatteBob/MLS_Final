import numpy as np
import pytest

from fedcrop.io import digest, emit, run_lock
from fedcrop.data import partition, validate_manifest, merge_duplicates
from fedcrop.privacy import PrivacyLedger, BudgetExhausted
from fedcrop.resources import make_devices, select_clients, assignment, workload
from fedcrop.federation import secure_round, validate_version, StaleUpdateError


def test_group_isolation_and_determinism():
    rows = [{"group_id": f"g{i//2}", "label": (i//2) % 10} for i in range(240)]
    first = partition(rows)
    assert first == partition(rows)
    owners = {}
    for client, indices in enumerate(first):
        assert indices
        for i in indices:
            key = rows[i]["group_id"]
            assert key not in owners or owners[key] == client
            owners[key] = client


def test_manifest_rejects_cross_split_duplicates():
    rows = [{"group_id": "g1", "sha256": "same", "split": "train"},
            {"group_id": "g2", "sha256": "same", "split": "test"}]
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest({"rows": rows, "manifest_hash": digest(rows)})


def test_near_duplicate_quarantine():
    rows = [{"leaf_id": "a", "sha256": "a", "dhash": 100, "original_split": "train", "source_path": "a"},
            {"leaf_id": "b", "sha256": "b", "dhash": 101, "original_split": "test", "source_path": "b"}]
    kept, removed = merge_duplicates(rows)
    assert not kept and sorted(removed) == ["a", "b"]


def test_ledger_persistence_and_budget(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = PrivacyLedger(path, "run", 0, 4, 1e-5, create=True)
    ledger.reserve(2, 0.1)
    spent = ledger.epsilon
    restored = PrivacyLedger(path, "run", 0, 4, 1e-5)
    assert restored.epsilon == spent and restored.entries == 1
    assert restored.epsilon == pytest.approx(restored.accountant.get_epsilon(1e-5), abs=1e-12)
    with pytest.raises(BudgetExhausted):
        restored.reserve(0.01, 1)
    assert restored.entries == 1
    with pytest.raises(FileNotFoundError):
        PrivacyLedger(tmp_path / "missing.json", "run", 0, 4, 1e-5)


def test_cached_rdp_matches_opacus_with_changed_workload(tmp_path):
    ledger = PrivacyLedger(tmp_path / "p.json", "run", 0, 100, 1e-5, create=True)
    for sigma, rate in [(2., .1), (2., .1), (3., .2), (2., .05)]:
        ledger.reserve(sigma, rate)
        assert ledger.epsilon == pytest.approx(ledger.accountant.get_epsilon(1e-5), abs=1e-12)


def test_fairness_and_no_feasible_model():
    devices = make_devices()
    devices[-1].age = 99
    feasible = {d.client_id: {"seconds": d.client_id + 1} for d in devices}
    selected = select_clients(devices, feasible, 6, "fair", np.random.default_rng(1))
    assert 11 in selected and len(set(selected)) == 6
    assert assignment([], devices[0]) is None


def test_battery_reserve_is_checked_before_work():
    device = make_devices()[4]
    device.battery = .200001
    profile = {"training": [{"device": "cpu", "batch_size": 8, "peak_rss_bytes": 1000, "step_seconds": 1.}]}
    config = {"batch_size": 8, "compute_scale": 1., "bandwidth_scale": 1., "secure_aggregation": True,
              "local_steps": 20, "round_deadline": 120., "battery_reserve": .2}
    assert workload(device, profile, config, 400000) is None


def test_telemetry_allowlist(tmp_path):
    with pytest.raises(ValueError):
        emit(tmp_path / "events.jsonl", private_loss=0.1)
    assert not (tmp_path / "events.jsonl").exists()


def test_single_writer_lock_and_release(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):
        pass


@pytest.mark.parametrize("drop_stage", ["setup", "share_keys", "collect_masked_vectors", "unmask"])
@pytest.mark.parametrize("drop_count", [0, 2, 3])
def test_real_secagg_recovery(drop_stage, drop_count):
    weights = [np.array([0.0, 0.0], dtype=np.float32)]
    called = []
    def train(client, parameters):
        called.append(client)
        return [np.array([client * .1, -client * .1], dtype=np.float32)]
    dropped = list(range(drop_count))
    result, stats = secure_round(weights, list(range(6)), train, 0, drop_ids=dropped, drop_stage=drop_stage)
    if drop_count >= 3 or (drop_count and drop_stage in {"setup", "share_keys"}):
        assert result is None
    else:
        contributors = list(range(6)) if drop_stage == "unmask" else list(range(drop_count, 6))
        mean = np.mean(contributors) * .1
        np.testing.assert_allclose(result[0], [mean, -mean], atol=1e-4)
    if drop_stage == "collect_masked_vectors":
        assert len(called) == 6
    assert stats["wire_bytes"] > 0


def test_stale_update_rejected():
    with pytest.raises(StaleUpdateError):
        validate_version(1, 2)


def test_aborted_aggregation_keeps_all_client_privacy(tmp_path):
    ledgers = [PrivacyLedger(tmp_path / f'{i}.json', 'run', i, 8, 1e-5, create=True) for i in range(6)]
    initial = [np.zeros(2, dtype=np.float32)]
    def train(client, parameters):
        ledgers[client].reserve(3., .1)
        return [np.ones(2, dtype=np.float32)]
    aggregate, _ = secure_round(initial, list(range(6)), train, 0, drop_ids=[0, 1, 2])
    assert aggregate is None
    np.testing.assert_array_equal(initial[0], np.zeros(2))
    for i in range(6):
        restored = PrivacyLedger(tmp_path / f'{i}.json', 'run', i, 8, 1e-5)
        assert restored.entries == 1 and restored.epsilon > 0

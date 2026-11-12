import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from fedcrop.data import merge_duplicates, inferred_leaf_id, leaf_id
from fedcrop.model import CropCNN, get_weights
from fedcrop.training import train_local
from fedcrop.privacy import PrivacyLedger
from fedcrop.warmup import load_public_init
from fedcrop.io import atomic_torch, atomic_json, digest, file_hash, read_json


def test_preserve_test_removes_training_side_of_transitive_component():
    rows = [{"leaf_id": "same", "sha256": "a", "original_split": "train", "source_path": "train1"},
            {"leaf_id": "same", "sha256": "b", "original_split": "train", "source_path": "train2"},
            {"leaf_id": "other", "sha256": "b", "original_split": "test", "source_path": "test"}]
    kept, removed = merge_duplicates(rows, preserve_test=True)
    assert [r["source_path"] for r in kept] == ["test"]
    assert set(removed) == {"train1", "train2"}


def test_mapping_alias_is_class_specific_and_inference_stays_unverified():
    mapping = {"com.g_fl 8310": ["Tomato___Target_Spot:::243.0"]}
    assert leaf_id("uuid___Com.G_TgS_FL 8310.JPG", "Tomato___Target_Spot", mapping)
    assert leaf_id("uuid___Com.G_FL 8310.JPG", "Tomato___Target", mapping) is None
    assert inferred_leaf_id("id___GH_HL Leaf 349.1.JPG", "Tomato___healthy") == inferred_leaf_id("id___GH_HL Leaf 349.JPG", "Tomato___healthy")
    assert inferred_leaf_id("id___GH_HL Leaf 349.JPG", "Tomato___healthy").startswith("unverified::")


def test_dp_head_training_keeps_public_features_and_spends_budget(tmp_path):
    torch.set_num_threads(2)
    original = get_weights(CropCNN(normalize=True))
    dataset = TensorDataset(torch.rand(16, 3, 96, 96), torch.arange(16) % 10)
    ledger = PrivacyLedger(tmp_path / "ledger.json", "head", 0, 8, 1e-5, create=True)
    config = {"device": "cpu", "learning_rate": .01, "clip_norm": 1., "physical_batch_size": 2,
              "freeze_features": True, "normalize": True}
    weights, stats = train_local(original, dataset, config, seed=3, steps=3, batch_size=4, ledger=ledger, sigma=3.)
    for before, after in zip(original[:-2], weights[:-2]):
        np.testing.assert_array_equal(before, after)
    assert not np.array_equal(original[-1], weights[-1])
    assert stats["steps"] == ledger.entries == 3 and 0 < ledger.epsilon <= 8


def test_public_checkpoint_rejects_different_pool_or_preprocessing(tmp_path):
    rows = [{"split": "public", "path": "a"}]
    manifest = {"rows": rows, "manifest_hash": "manifest"}
    path = tmp_path / "init.pt"
    atomic_torch(path, {"state_dict": {}, "provenance": {"manifest_hash": "manifest", "public_rows_hash": digest(rows), "normalize": True}})
    load_public_init(path, manifest, True)
    with pytest.raises(ValueError):
        load_public_init(path, manifest, False)
    with pytest.raises(ValueError):
        load_public_init(path, {**manifest, "rows": []}, True)


def test_public_warmup_never_trains_or_evaluates_private_test_rows(tmp_path, monkeypatch):
    import fedcrop.warmup as warmup
    rows = []
    for index, split in enumerate(["public", "public", "train", "validation", "test"]):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(bytes([index]))
        rows.append({"path": str(path), "sha256": file_hash(path), "split": split,
                     "group_id": str(index), "label": index % 2})
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, {"rows": rows, "manifest_hash": digest(rows)})
    accessed = []
    class CheckedDataset:
        def __init__(self, rows, augment=False):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            split = self.rows[i]["split"]
            assert split in {"public", "validation"}
            accessed.append(split)
            return torch.zeros(3, 96, 96), self.rows[i]["label"]
    monkeypatch.setattr(warmup, "ImageDataset", CheckedDataset)
    warmup.public_warmup({"manifest": str(manifest_path), "seed": 1, "device": "cpu"}, tmp_path / "public_run", epochs=1)
    assert set(accessed) == {"public", "validation"}
    assert read_json(tmp_path / "public_run/summary.json")["test_evaluated"] is False

"""Pinned source download, conservative leaf/duplicate grouping, fixed partitions."""
from __future__ import annotations

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import time
import re

import numpy as np
import requests
from PIL import Image

from .io import atomic_json, read_json, file_hash, digest

CLASSES = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
           "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
           "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
           "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
           "Tomato___healthy"]


def fetch(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(20, 120))
            response.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(response.content)
            tmp.replace(dest)
            return dest
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def image_identifier(filename):
    identifier = filename.replace("_final_masked", "").split("___")[-1].split("copy")[0]
    identifier = identifier.replace(".JPG", "").replace(".jpg", "").replace(".PNG", "").replace(".png", "").strip().lower()
    return identifier


def leaf_id(filename, label, leaf_map):
    identifier = image_identifier(filename)
    suggestions = leaf_map.get(identifier, [])
    if not suggestions and label == "Tomato___Target_Spot":
        # Source CSV uses Com.G_FL where the image names use Com.G_TgS_FL.
        suggestions = leaf_map.get(identifier.replace("com.g_tgs_fl", "com.g_fl"), [])
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    matching = [s for s in suggestions if s.split("::", 1)[0] == label]
    if len(matching) == 1:
        return matching[0]
    return None  # Never pretend an unverified filename is a known physical leaf.


def inferred_leaf_id(filename, label):
    """Conservative filename grouping, never a verified biological identity."""
    identifier = image_identifier(filename)
    match = re.fullmatch(r"(.+?\sleaf\s*\d+)(?:\.\d+)?", identifier)
    return f"unverified::{label}::{match.group(1) if match else identifier}"


def prepare(root="data", max_groups_per_class=0, seed=2026, workers=8, policy="strict",
            train_images_per_class=0, test_images_per_class=0, preserve_test=False,
            manifest_name="manifest.json"):
    if policy not in {"strict", "audited"}:
        raise ValueError("Grouping policy must be strict or audited")
    root = Path(root)
    metadata = root / "source"
    metadata.mkdir(parents=True, exist_ok=True)
    source_path = metadata / "revision.json"
    if source_path.exists():
        source = read_json(source_path)
    else:
        hf = requests.get("https://huggingface.co/api/datasets/mohanty/PlantVillage", timeout=60)
        hf.raise_for_status()
        # A verified immutable upstream commit avoids unauthenticated GitHub API limits.
        source = {"hf_revision": hf.json()["sha"],
                  "github_revision": "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"}
        atomic_json(source_path, source)
    base = f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{source['hf_revision']}"
    for name in ["leaf_grouping/leaf-map.json", "splits/color_train.txt", "splits/color_test.txt"]:
        fetch(f"{base}/{name}", metadata / name)
    leaf_map = read_json(metadata / "leaf_grouping/leaf-map.json")
    groups = defaultdict(list)
    unknown = []
    for split in ["train", "test"]:
        for rel in (metadata / f"splits/color_{split}.txt").read_text().splitlines():
            parts = rel.strip().split("/")
            if len(parts) != 4 or parts[:2] != ["raw", "color"] or parts[2] not in CLASSES:
                continue
            if "\\" in parts[3] or parts[3] in {".", "..", ""}:
                raise ValueError("Unsafe filename in upstream split metadata")
            group = leaf_id(parts[3], parts[2], leaf_map)
            verified = group is not None
            if group is None:
                unknown.append(rel)
                if policy == "strict":
                    continue
                group = inferred_leaf_id(parts[3], parts[2])
            groups[group].append({"source_path": rel, "label": CLASSES.index(parts[2]),
                                  "leaf_id": group, "leaf_verified": verified, "original_split": split})
    rng = np.random.default_rng(seed)
    selected = []
    source_quarantine = []
    if preserve_test:
        for group, rr in list(groups.items()):
            if len({r["original_split"] for r in rr}) > 1:
                source_quarantine.extend(r["source_path"] for r in rr if r["original_split"] == "train")
                groups[group] = [r for r in rr if r["original_split"] == "test"]
    for split in ["train", "test"]:
        for label in range(10):
            keys = sorted(g for g, rows in groups.items()
                          if rows[0]["label"] == label and rows[0]["original_split"] == split)
            rng.shuffle(keys)
            if max_groups_per_class:
                keys = keys[:max_groups_per_class]
            budget = train_images_per_class if split == "train" else test_images_per_class
            count = 0
            for key in keys:
                if budget and count >= budget:
                    break
                selected.extend(groups[key])
                count += len(groups[key])
    if not selected:
        raise ValueError("No verified tomato leaf groups found; inspect source metadata.")

    def download(row):
        rel = row["source_path"]
        dest = root / "images" / rel
        fetch(f"https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/{source['github_revision']}/{quote(rel)}", dest)
        row["path"] = dest.resolve().as_posix()
        row["sha256"] = file_hash(dest)
        with Image.open(dest) as im:
            arr = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
            row["dhash"] = int.from_bytes(np.packbits(arr[:, 1:] > arr[:, :-1]).tobytes(), "big")
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(download, selected)):
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"Downloaded/verified {i + 1}/{len(selected)} images", flush=True)
    rows, quarantined = merge_duplicates(rows, preserve_test=preserve_test)
    quarantined = source_quarantine + quarantined
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    for label in range(10):
        keys = sorted(g for g, rr in by_group.items()
                      if rr[0]["original_split"] == "train" and rr[0]["label"] == label)
        rng.shuffle(keys)
        n = max(1, round(len(keys) * 0.1))
        if len(keys) < 4:
            if policy == "strict" and not keys:
                continue
            raise ValueError(f"Too few groups in class {CLASSES[label]}")
        for i, key in enumerate(keys):
            split = "validation" if i < n else "public" if i < 2 * n else "train"
            for row in by_group[key]:
                row["split"] = split
    for row in rows:
        if row["original_split"] == "test":
            row["split"] = "test"
    result = {"schema": 1, "source": source, "classes": CLASSES, "seed": seed, "grouping_policy": policy,
              "all_physical_leaves_verified": all(r["leaf_verified"] for r in rows),
              "verified_images": sum(r["leaf_verified"] for r in rows),
              "present_labels": sorted({r["label"] for r in rows}),
              "max_groups_per_class_per_source_split": max_groups_per_class,
              "image_budgets": {"train": train_images_per_class, "test": test_images_per_class},
              "unknown_leaf_paths": unknown, "quarantined_paths": quarantined,
              "duplicate_policy": "Known/inferred groups + SHA256 + dHash <= 2; " + ("cross-source training rows removed; official test preserved" if preserve_test else "cross-source components quarantined"),
              "counts": dict(Counter(r["split"] for r in rows)), "rows": rows}
    result["manifest_hash"] = digest(rows)
    validate_manifest(result, verify_files=True)
    atomic_json(root / manifest_name, result)
    print(result["counts"], flush=True)
    return result


def merge_duplicates(rows, preserve_test=False):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    seen = {}
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        for key in [("leaf", row["leaf_id"]), ("hash", row["sha256"])]:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
        # With <=2 changed bits, at least one of these three segments is unchanged.
        h = row.get("dhash")
        if h is None:
            continue
        keys = [(k, (h >> shift) & ((1 << width) - 1))
                for k, (shift, width) in enumerate([(0, 22), (22, 21), (43, 21)])]
        candidates = {j for key in keys for j in buckets[key]}
        for j in candidates:
            if (h ^ rows[j]["dhash"]).bit_count() <= 2:
                union(i, j)
        for key in keys:
            buckets[key].append(i)
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    kept, quarantined = [], []
    for group in components.values():
        if len({r["original_split"] for r in group}) > 1:
            quarantined.extend(r["source_path"] for r in group if not preserve_test or r["original_split"] == "train")
            if not preserve_test:
                continue
            group = [r for r in group if r["original_split"] == "test"]
        gid = digest(sorted({r["leaf_id"] for r in group}))[:24]
        for row in group:
            row["group_id"] = gid
            kept.append(row)
    return kept, quarantined


def validate_manifest(manifest, verify_files=False):
    splits, hashes = {}, {}
    for row in manifest["rows"]:
        split = row["split"]
        if split not in {"train", "validation", "public", "test"}:
            raise ValueError("Invalid data split")
        for store, key in [(splits, row["group_id"]), (hashes, row["sha256"])]:
            if key in store and store[key] != split:
                raise ValueError("Group or duplicate leakage across splits")
            store[key] = split
        if verify_files and file_hash(row["path"]) != row["sha256"]:
            raise ValueError(f"Checksum mismatch: {row['path']}")
    if manifest.get("manifest_hash") != digest(manifest["rows"]):
        raise ValueError("Manifest content hash mismatch")


def partition(rows, clients=12, alpha=0.3, seed=2026):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["group_id"]].append(i)
    if len(groups) < clients:
        raise ValueError("Not enough groups for nonempty clients")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        probabilities = (np.full((10, clients), 1 / clients) if alpha is None
                         else rng.dirichlet(np.full(clients, alpha), 10))
        assigned = [[] for _ in range(clients)]
        for gid in sorted(groups):
            indices = groups[gid]
            label = Counter(rows[i]["label"] for i in indices).most_common(1)[0][0]
            owner = int(rng.choice(clients, p=probabilities[label]))
            assigned[owner].extend(indices)
        if all(assigned):
            return assigned
    raise ValueError("Could not construct nonempty group-preserving partition")


class ImageDataset:
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            arr = np.array(image.convert("RGB").resize((96, 96)), dtype=np.float32) / 255.0
        if self.augment:
            arr = np.rot90(arr, int(torch.randint(4, (1,))))
            if bool(torch.rand(1) < 0.5):
                arr = np.fliplr(arr)
        return torch.from_numpy(arr.transpose(2, 0, 1).copy()), int(row["label"])

"""Derive a mapping-verified sensitivity subset without changing existing splits."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedcrop.io import read_json, atomic_json, digest
from fedcrop.data import validate_manifest

original = read_json("data/manifest.json")
groups = defaultdict(list)
for row in original["rows"]:
    groups[row["group_id"]].append(row)
rows = [row for group in groups.values() if all(r["leaf_verified"] for r in group) for row in group]
subset = {**original, "rows": rows, "grouping_policy": "strict-derived",
          "parent_manifest_hash": original["manifest_hash"], "manifest_hash": digest(rows),
          "all_physical_leaves_verified": True, "verified_images": len(rows),
          "counts": dict(Counter(r["split"] for r in rows)),
          "present_labels": sorted({r["label"] for r in rows})}
validate_manifest(subset, verify_files=True)
atomic_json("data/manifest_verified.json", subset)
print({"counts": subset["counts"], "present_labels": subset["present_labels"]})

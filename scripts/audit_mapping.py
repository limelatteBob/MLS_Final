"""Reproducible source mapping coverage; no invented physical leaf identities."""
import sys
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedcrop.data import CLASSES, image_identifier, leaf_id
from fedcrop.io import read_json, atomic_json


def main():
    mapping = read_json("data/source/leaf_grouping/leaf-map.json")
    counts = {c: Counter() for c in CLASSES}
    missing = defaultdict(Counter)
    groups = defaultdict(list)
    review_rows = []
    for split in ["train", "test"]:
        for path in Path(f"data/source/splits/color_{split}.txt").read_text().splitlines():
            parts = path.split("/")
            if len(parts) != 4 or parts[2] not in CLASSES:
                continue
            label, name = parts[2:]
            group = leaf_id(name, label, mapping)
            counts[label]["total"] += 1
            counts[label]["verified" if group else "missing"] += 1
            if group:
                groups[group].append((split, label))
                if label == "Tomato___Target_Spot" and not mapping.get(image_identifier(name)):
                    counts[label]["alias_recovered"] += 1
            else:
                prefix = re.sub(r"[\s_]*\d.*$", "", image_identifier(name))
                missing[label][prefix] += 1
                review_rows.append({"source_path": path, "class": label, "source_split": split,
                                    "prefix": prefix, "verified_leaf_id": "", "evidence_reference": ""})
    cross_train = Counter()
    for rr in groups.values():
        if len({s for s, c in rr}) > 1:
            cross_train.update(c for s, c in rr if s == "train")
    manifests = {}
    for name in ["manifest.json", "manifest_v2.json"]:
        path = Path("data") / name
        if not path.exists():
            continue
        manifest = read_json(path)
        manifests[name] = {"hash": manifest["manifest_hash"], "counts": manifest["counts"],
                           "quarantined_images": len(manifest["quarantined_paths"]),
                           "verified_images": manifest["verified_images"],
                           "unverified_images": sum(not r["leaf_verified"] for r in manifest["rows"]),
                           "classes": {s: {CLASSES[i]: n for i, n in sorted(Counter(r["label"] for r in manifest["rows"] if r["split"] == s).items())}
                                       for s in ["train", "validation", "public", "test"]}}
        proxy = defaultdict(list)
        for row in manifest["rows"]:
            if not row["leaf_verified"]:
                proxy[row["leaf_id"]].append(row)
        manifests[name]["unverified_proxy_groups"] = len(proxy)
        manifests[name]["unverified_groups_with_multiple_images"] = sum(len(rr) > 1 for rr in proxy.values())
    result = {"source": read_json("data/source/revision.json"),
              "classes": {c: {**dict(counts[c]), "missing_prefixes": dict(missing[c]),
                               "source_train_rows_sharing_verified_test_leaf": cross_train[c]} for c in CLASSES},
              "manifests": manifests,
              "interpretation": "Missing prefixes identify absent source batches, not evidence for guessing biological leaf IDs. Filename leaf-number groups remain unverified."}
    atomic_json("artifacts/mapping_audit_v2.json", result)
    with Path("artifacts/unmapped_leaf_review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_path", "class", "source_split", "prefix", "verified_leaf_id", "evidence_reference"])
        writer.writeheader()
        writer.writerows(review_rows)
    print({c: dict(counts[c]) for c in CLASSES})


if __name__ == "__main__":
    main()

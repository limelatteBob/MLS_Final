"""Comparable configuration generation and honest aggregation of completed runs."""
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, atomic_json


def experiment_suite(config_path, output, execute=False, stress=False):
    from .experiment import load_config, run
    base = load_config(config_path)
    root = Path(output)
    variants = {
        "local": {"mode": "local", "dp": False, "secure_aggregation": False, "export_after_training": False},
        "centralized": {"mode": "centralized", "dp": False, "secure_aggregation": False, "export_after_training": False},
        "uniform_nonprivate": {"dp": False, "secure_aggregation": False, "scheduler": "uniform", "export_after_training": False},
        "uniform_dp": {"dp": True, "secure_aggregation": False, "scheduler": "uniform", "export_after_training": False},
        "uniform_private": {"dp": True, "secure_aggregation": True, "scheduler": "uniform"},
        "speed_private": {"dp": True, "secure_aggregation": True, "scheduler": "speed"},
        "full": {"dp": True, "secure_aggregation": True, "scheduler": "fair"},
    }
    if stress:
        for name, change in {"iid": {"alpha": None}, "skew01": {"alpha": .1},
                             "drop0": {"dropout": 0}, "drop30": {"dropout": .3},
                             "epsilon4": {"epsilon_cap": 4}, "compute2": {"compute_scale": 2},
                             "bandwidth_half": {"bandwidth_scale": .5}}.items():
            variants[name] = {**variants["full"], **change}
    experiments = []
    for name, changes in variants.items():
        for seed in [42, 43, 44]:
            config = {**base, **changes, "seed": seed, "output": str(root / f"{name}_seed{seed}")}
            path = root / "configs" / f"{name}_seed{seed}.json"
            atomic_json(path, config)
            experiments.append({"variant": name, "seed": seed, "config": str(path), "output": config["output"]})
    atomic_json(root / "suite.json", {"experiments": experiments, "target_f1": base["target_f1"],
                                     "note": "Freeze target_f1 from pilot validation before formal comparisons. Null is not a reached target."})
    print(f"Prepared {len(experiments)} experiment configurations in {root}", flush=True)
    if execute:
        for experiment in experiments:
            config = read_json(experiment["config"])
            result_dir = Path(config["output"])
            if (result_dir / "summary.json").exists():
                continue
            print(f"Running {result_dir.name}", flush=True)
            run(config, resume=(result_dir / "state.pt").exists())


def report(root, output):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, compression = [], []
    privacy_records, participation_records = [], []
    fig, ax = plt.subplots(figsize=(10, 5))
    for path in sorted(root.rglob("summary.json")):
        if not (path.parent / "config.json").exists():
            continue  # Public initialization artifacts are not federated experiments.
        summary = read_json(path)
        config = read_json(path.parent / "config.json")
        variant = path.parent.name.rsplit("_seed", 1)[0]
        test_f1 = (summary.get("test") or {}).get("macro_f1", summary.get("mean_test_f1"))
        if (path.parent / "final_test.json").exists():
            test_f1 = read_json(path.parent / "final_test.json")["test"]["macro_f1"]
        if "test" not in summary and summary.get("models"):
            from .model import metrics_from_confusion
            test_f1 = metrics_from_confusion(sum(np.asarray(r["test"]["confusion_matrix"]) for r in summary["models"]))["macro_f1"]
        records.append({"run": path.parent.name, "variant": variant, "seed": config["seed"],
                        "manifest_hash": summary.get("manifest_hash"), "test_macro_f1": test_f1,
                        "simulated_seconds": summary.get("simulated_seconds"), "wire_bytes": summary.get("wire_bytes"),
                        "abort_rate": summary.get("abort_rate"), "worst_quartile_f1": summary.get("worst_quartile_f1"),
                        "time_to_target": summary.get("time_to_target"), "target_status": summary.get("target_status", "not_applicable")})
        for client, epsilon in summary.get("epsilon_by_client", {}).items():
            privacy_records.append({"run": path.parent.name, "client_id": int(client), "epsilon": epsilon,
                                    "cap": config["epsilon_cap"]})
        for device in summary.get("devices", []):
            participation_records.append({"run": path.parent.name, **device})
        rounds = path.parent / "rounds.json"
        if rounds.exists():
            history = read_json(rounds)
            ax.plot([r["simulated_seconds"] for r in history], [r["macro_f1"] for r in history], label=path.parent.name)
        registry = path.parent / "registry.json"
        if registry.exists():
            for model in read_json(registry)["models"]:
                compression.append({"run": path.parent.name, "model": model["name"],
                                    "size_bytes": model["artifact_bytes"], "p95_ms": model["p95_ms"],
                                    "validation_macro_f1": model["validation"]["macro_f1"],
                                    "compression_ratio": model["compression_ratio"],
                                    "target_met": model["compression_target_met"]})
    ax.set(xlabel="Simulated parallel time (seconds)", ylabel="Validation macro-F1", title="FedCrop convergence")
    if ax.lines:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)
    table = pd.DataFrame(records)
    table.to_csv(output / "runs.csv", index=False)
    if len(table):
        metrics = ["test_macro_f1", "simulated_seconds", "wire_bytes", "abort_rate", "worst_quartile_f1", "time_to_target"]
        for metric in metrics:
            table[metric] = pd.to_numeric(table[metric], errors="coerce")
        # Different data manifests must not be silently pooled.
        grouped = table.groupby(["variant", "manifest_hash"], dropna=False)[metrics].agg(["mean", "std", "count"])
        grouped.to_csv(output / "mean_std.csv")
    pd.DataFrame(compression).to_csv(output / "compression.csv", index=False)
    pd.DataFrame(privacy_records).to_csv(output / "privacy.csv", index=False)
    pd.DataFrame(participation_records).to_csv(output / "participation.csv", index=False)
    for name, points, value, label in [("privacy", privacy_records, "epsilon", "Spent epsilon"),
                                       ("participation", participation_records, "participations", "Committed participation count")]:
        if points:
            frame = pd.DataFrame(points)
            matrix = frame.pivot(index="run", columns="client_id", values=value)
            fig, ax = plt.subplots(figsize=(10, max(4, len(matrix) * .3)))
            heatmap = ax.imshow(matrix.values, aspect="auto", cmap="viridis", vmin=0)
            ax.set_xticks(range(len(matrix.columns)), matrix.columns)
            ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
            ax.set(xlabel="Client ID", title=name.capitalize())
            fig.colorbar(heatmap, ax=ax, label=label)
            fig.tight_layout()
            fig.savefig(output / f"{name}.png", dpi=160)
            plt.close(fig)
    if compression:
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in compression:
            ax.scatter(row["size_bytes"] / 1024, row["validation_macro_f1"])
            ax.annotate(row["model"], (row["size_bytes"] / 1024, row["validation_macro_f1"]), fontsize=7)
        ax.set(xlabel="ONNX artifact (KiB)", ylabel="Validation macro-F1", title="Compression tradeoffs")
        fig.tight_layout()
        fig.savefig(output / "compression.png", dpi=160)
        plt.close(fig)
    atomic_json(output / "coverage.json", {"completed_runs": len(records),
                "note": "A one-seed run has undefined sample standard deviation. Missing targets remain missing. Smoke results are not full evaluation."})
    print(f"Collected {len(records)} completed runs into {output}")

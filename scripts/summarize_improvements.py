"""Derive tables and publication-ready figures from frozen measured artifacts."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fedcrop.io import read_json, atomic_json


if __name__ == "__main__":
    destination = Path("artifacts/improvements")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ["improved_legacy", "improved_seed42", "improved_seed43", "improved_seed44", "improved_nonprivate"]:
        directory = Path("runs") / name
        if not (directory / "final_test.json").exists():
            continue
        summary, final = read_json(directory / "summary.json"), read_json(directory / "final_test.json")
        records.append({"run": name, "manifest_hash": summary["manifest_hash"],
                        "initial_validation_f1": summary["initial_validation_f1"],
                        "validation_f1": summary["best_validation_f1"], "best_round": summary["best_round"],
                        "rounds": summary["rounds"], "test_f1": final["test"]["macro_f1"],
                        "public_test_f1": final["public_only_test"]["macro_f1"],
                        "test_accuracy": final["test"]["accuracy"],
                        "participating_clients": sum(d["participations"] > 0 for d in summary["devices"]),
                        "max_epsilon": max(summary["epsilon_by_client"].values(), default=0)})
    pd.DataFrame(records).to_csv(destination / "comparison.csv", index=False)
    private = [r for r in records if r["run"].startswith("improved_seed")]
    stats = {"runs": records, "scope": "Shared public seed42 initialization; conditional FL seed variability"}
    if private:
        scores = [r["test_f1"] for r in private]
        stats["private_test_mean"] = float(np.mean(scores))
        stats["private_test_std"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else None
    atomic_json(destination / "summary.json", stats)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ["public_legacy", "public_v2"]:
        summary = read_json(Path("runs") / name / "summary.json")
        axes[0].plot([r["epoch"] for r in summary["history"]], [r["validation_f1"] for r in summary["history"]], label=name)
    axes[0].set(xlabel="Public pretraining epoch", ylabel="Validation macro-F1", title="Separate data manifests")
    axes[0].legend()
    for r in records:
        if r["run"] == "improved_legacy":
            continue
        history = read_json(Path("runs") / r["run"] / "rounds.json")
        axes[1].plot([0] + [x["round"] for x in history], [r["initial_validation_f1"]] + [x["macro_f1"] for x in history], label=r["run"])
    axes[1].set(xlabel="Federated round", ylabel="Validation macro-F1", title="V2: same public initialization")
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "validation_curves.png", dpi=180)
    plt.close(fig)
    selected = Path("runs/improved_seed42/final_test.json")
    if selected.exists():
        final = read_json(selected)
        matrix = np.asarray(final["test"]["confusion_matrix"])
        normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        labels = ["Bacterial", "Early blight", "Late blight", "Leaf Mold", "Septoria", "Spider mites", "Target Spot", "Yellow Curl", "Mosaic", "Healthy"]
        ax.set(xticks=range(10), yticks=range(10), xticklabels=labels, yticklabels=labels,
               xlabel="Predicted", ylabel="Actual", title="V2 protected model, seed42: test recall by class")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for i in range(10):
            for j in range(10):
                if normalized[i, j] >= .05:
                    ax.text(j, i, f"{normalized[i,j]:.2f}", ha="center", va="center", color="white" if normalized[i,j] > .55 else "black", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(destination / "confusion_seed42.png", dpi=180)
        plt.close(fig)
    lines = ["# 改进实验实测结果", "", "三个隐私运行共享一个公共初始化；标准差只表示联邦阶段随机性。旧划分单列。", "",
             "| 运行 | 初始验证 F1 | 最佳验证 F1 | 测试 F1 | 公共模型测试 F1 | 最佳轮次 | 参与客户端 | 最大 epsilon |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['run']} | {r['initial_validation_f1']:.4f} | {r['validation_f1']:.4f} | {r['test_f1']:.4f} | {r['public_test_f1']:.4f} | {r['best_round']} | {r['participating_clients']} | {r['max_epsilon']:.3f} |")
    if private:
        lines.extend(["", f"新版隐私流程测试 macro-F1：{stats['private_test_mean']:.4f}，样本标准差 {stats['private_test_std']:.4f}（n={len(private)}）。"] if len(private)>1 else [])
    lines.extend(["", "![验证曲线](../artifacts/improvements/validation_curves.png)", "", "![分类混淆](../artifacts/improvements/confusion_seed42.png)", "",
                  "最差类别、公共基线收益和最佳轮次应与总分一起解读。这里只计算固定流程的结果，不根据测试集重新选择超参数。"])
    registry_path = Path("runs/improved_seed42/registry.json")
    if registry_path.exists():
        registry = read_json(registry_path)
        lines.extend(["", "## 导出与压缩", "", "同一个 seed42 教师，量化与蒸馏仅访问公共池，是否可部署按验证 F1 相对教师下降不超过 3 个百分点决定。",
                      "", "| 模型 | 验证 F1 | 测试 F1 | 大小 KiB | 压缩倍数 | CPU p95 ms | 质量门槛 |",
                      "|---|---:|---:|---:|---:|---:|---|"])
        for m in registry["models"]:
            lines.append(f"| {m['name']} | {m['validation']['macro_f1']:.4f} | {m['test']['macro_f1']:.4f} | {m['artifact_bytes']/1024:.1f} | {m['compression_ratio']:.2f} | {m['p95_ms']:.3f} | {'通过' if m['quality_acceptable'] else '不通过，不分配'} |")
        distillation = Path("runs/improved_seed42/models/distillation.json")
        if distillation.exists():
            d = read_json(distillation)
            lines.extend(["", f"学生模型预算 {d['epochs_budget']} epochs，实际 {len(d['history'])} epochs；按验证集选最佳检查点，公共增强为 {d['augment']}。"])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([r["epoch"] for r in d["history"]], [r["validation_f1"] for r in d["history"]], label="Student FP32 before quantization")
            ax.axhline(registry["models"][0]["validation"]["macro_f1"], color="black", linestyle="--", label="Teacher")
            ax.set(xlabel="Distillation epoch", ylabel="Validation macro-F1", title="Public-pool student training")
            ax.legend()
            fig.tight_layout()
            fig.savefig(destination / "distillation.png", dpi=180)
            plt.close(fig)
            lines.extend(["", "![学生模型训练](../artifacts/improvements/distillation.png)"])
    Path("docs/IMPROVEMENT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(stats)

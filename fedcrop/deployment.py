"""Public-pool compression and measured, resource-constrained model assignment."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .benchmark import image_array
from .data import ImageDataset
from .io import atomic_json, read_json, file_hash, emit
from .model import CropCNN, evaluate
from .resources import make_devices, assignment, Device
from .training import seed_everything


class PublicCalibration(CalibrationDataReader):
    def __init__(self, rows):
        self.rows = rows
        self.rewind()

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = ({"images": image_array(r["path"])} for r in self.rows)


def distill(teacher, public, validation, config):
    seed_everything(config["seed"])
    device = config["device"]
    teacher.to(device).eval()
    student = CropCNN((16, 32, 64), normalize=config.get("normalize", False)).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.001, weight_decay=0.001)
    best_score, best, bad, history = -1, None, 0, []
    for epoch in range(config["distill_epochs"]):
        student.train()
        for x, y in DataLoader(public, batch_size=32, shuffle=True):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                soft = torch.softmax(teacher(x) / 3, dim=1)
            logits = student(x)
            loss = 0.5 * torch.nn.functional.cross_entropy(logits, y) + 4.5 * torch.nn.functional.kl_div(
                torch.log_softmax(logits / 3, dim=1), soft, reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score = evaluate(student, validation, device)["macro_f1"]
        if score > best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            bad = 0
        else:
            bad += 1
        history.append({"epoch": epoch + 1, "validation_f1": score})
        print(f"Student epoch {epoch + 1}: validation F1={score:.4f}, best={best_score:.4f}", flush=True)
        if epoch + 1 >= 30 and bad >= config.get("distill_patience", 20):
            break
    if best is not None:
        student.load_state_dict(best)
    student.distillation_history = history
    return student.cpu().eval()


def export_onnx(model, target, sample):
    model.cpu().eval()
    torch.onnx.export(model, torch.from_numpy(sample), str(target), input_names=["images"],
                      output_names=["logits"], dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(torch.from_numpy(sample)).numpy()
    np.testing.assert_allclose(session.run(None, {"images": sample})[0], expected, atol=1e-5, rtol=1e-4)


def export_models(config, run_dir):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if config["distill_epochs"] < 1:
        raise ValueError("distill_epochs must be positive")
    run_dir = Path(run_dir)
    export_dir = run_dir / "models"
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(config["manifest"])
    public_rows = [r for r in manifest["rows"] if r["split"] == "public"]
    val = ImageDataset([r for r in manifest["rows"] if r["split"] == "validation"])
    if not public_rows:
        raise ValueError("Export requires an explicit public calibration pool")
    teacher = CropCNN(normalize=config.get("normalize", False))
    teacher.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    sample = image_array(public_rows[0]["path"])
    fp32 = export_dir / "teacher_fp32.onnx"
    export_onnx(teacher, fp32, sample)
    student = distill(teacher, ImageDataset(public_rows, augment=config.get("distill_augment", False)), val, config)
    atomic_json(export_dir / "distillation.json", {"manifest_hash": manifest["manifest_hash"],
                "teacher_sha256": file_hash(run_dir / "best.pt"), "epochs_budget": config["distill_epochs"],
                "augment": config.get("distill_augment", False), "seed": config["seed"],
                "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .001,
                "history": student.distillation_history, "selection": "validation macro-F1 only"})
    student_fp32 = export_dir / "student_fp32.onnx"
    export_onnx(student, student_fp32, sample)
    variants = [("teacher_fp32", fp32)]
    failures = []
    for name, source in [("teacher_int8", fp32), ("student_int8", student_fp32)]:
        target = export_dir / f"{name}.onnx"
        try:
            processed = export_dir / f"{name}_preprocessed.onnx"
            quant_pre_process(str(source), str(processed))
            quantize_static(str(processed), str(target), PublicCalibration(public_rows),
                            quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                            weight_type=QuantType.QInt8, per_channel=True)
            onnx.checker.check_model(str(target))
            variants.append((name, target))
        except (RuntimeError, ValueError) as exc:
            failures.append({"model": name, "reason": str(exc)})
    registry = []
    for name, target in variants:
        metrics_path = export_dir / f"{name}_benchmark.json"
        try:
            subprocess.run([sys.executable, "-m", "fedcrop.benchmark", str(target), str(config["manifest"]),
                            str(metrics_path), "--iterations", str(config["benchmark_iterations"])], check=True)
        except subprocess.CalledProcessError as exc:
            if name == "teacher_fp32":
                raise
            failures.append({"model": name, "reason": str(exc)})
            continue
        registry.append({"name": name, "path": target.resolve().as_posix(), "sha256": file_hash(target),
                         "artifact_bytes": target.stat().st_size, "teacher_sha256": file_hash(run_dir / "best.pt"),
                         **read_json(metrics_path)})
    base = registry[0]
    for model in registry:
        model["compression_ratio"] = base["artifact_bytes"] / model["artifact_bytes"]
        model["validation_f1_loss"] = base["validation"]["macro_f1"] - model["validation"]["macro_f1"]
        model["quality_acceptable"] = model["validation_f1_loss"] <= 0.03
        model["compression_target_met"] = model["compression_ratio"] >= 3 and model["quality_acceptable"]
    result = {"schema": 1, "manifest_hash": manifest["manifest_hash"], "teacher_is_dp": config["dp"],
              "models": registry, "failures": failures}
    atomic_json(run_dir / "registry.json", result)
    assign_models(run_dir)
    return result


def assign_models(run_dir, device_file=None, latency_budget_ms=100):
    run_dir = Path(run_dir)
    registry = read_json(run_dir / "registry.json")["models"]
    config = read_json(run_dir / "config.json")
    devices = ([Device(**d) for d in read_json(device_file)] if device_file
               else make_devices(config["num_clients"], config["partition_seed"]))
    path = run_dir / "assignments.json"
    previous_document = read_json(path) if path.exists() else {}
    previous = previous_document.get("assignments", {})
    previous_hashes = previous_document.get("model_sha256_by_client", {})
    selected, versions, changes, download_bytes = {}, {}, [], 0
    for d in devices:
        name = assignment([m for m in registry if m.get("quality_acceptable", True)], d,
                          latency_budget_ms=latency_budget_ms, compute_scale=config["compute_scale"])
        selected[str(d.client_id)] = name
        model_hash = next((m.get("sha256", m["name"]) for m in registry if m["name"] == name), None)
        versions[str(d.client_id)] = model_hash
        if previous.get(str(d.client_id)) != name or previous_hashes.get(str(d.client_id)) != model_hash:
            amount = next((m["artifact_bytes"] for m in registry if m["name"] == name), 0)
            download_bytes += amount
            changes.append({"client_id": d.client_id, "from": previous.get(str(d.client_id)), "to": name,
                            "model_sha256": model_hash})
            emit(run_dir / "events.jsonl", event="assignment", client_id=d.client_id,
                 assigned_model=name, download_bytes=amount, status="ready" if name else "paused")
    result = {"assignments": selected, "changes": changes, "download_bytes": download_bytes,
              "model_sha256_by_client": versions,
              "latency_budget_ms": latency_budget_ms, "devices": [asdict(d) for d in devices]}
    atomic_json(path, result)
    return result


def predict(run_dir, image_path, client_id=0):
    from .data import CLASSES
    run_dir = Path(run_dir)
    selected = read_json(run_dir / "assignments.json")["assignments"].get(str(client_id))
    if selected is None:
        raise RuntimeError("Inference paused: this device has no model satisfying its budgets")
    model = next(m for m in read_json(run_dir / "registry.json")["models"] if m["name"] == selected)
    if file_hash(model["path"]) != model["sha256"]:
        raise ValueError("Deployment artifact checksum mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(model["path"], sess_options=options, providers=["CPUExecutionProvider"])
    logits = session.run(None, {"images": image_array(image_path)})[0][0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return {"client_id": client_id, "model": selected, "prediction": CLASSES[int(probabilities.argmax())],
            "probabilities": {label: float(value) for label, value in zip(CLASSES, probabilities)}}

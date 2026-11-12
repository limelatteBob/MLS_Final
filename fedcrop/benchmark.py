"""Isolated ONNX CPU inference measurement (does not import PyTorch)."""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

from .io import read_json, atomic_json


def image_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((96, 96)), dtype=np.float32).transpose(2, 0, 1)[None] / 255.0


def benchmark(model_path, manifest_path, iterations=100):
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.002):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        rows = read_json(manifest_path)["rows"]
        calibration = [r for r in rows if r["split"] == "public"]
        sample_x = image_array(calibration[0]["path"])
        for _ in range(10):
            session.run(None, {"images": sample_x})
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            session.run(None, {"images": sample_x})
            times.append((time.perf_counter_ns() - start) / 1e6)
        evaluations = {}
        for split in ["validation", "test"]:
            matrix = np.zeros((10, 10), dtype=np.int64)
            for row in rows:
                if row["split"] == split:
                    predicted = int(session.run(None, {"images": image_array(row["path"])})[0].argmax(1)[0])
                    matrix[row["label"], predicted] += 1
            tp = matrix.diagonal()
            denominator = matrix.sum(0) + matrix.sum(1)
            evaluations[split] = {"macro_f1": float(np.divide(2 * tp, denominator, out=np.zeros(10), where=denominator != 0).mean()),
                                  "accuracy": float(tp.sum() / max(1, matrix.sum())),
                                  "per_class_recall": np.divide(tp, matrix.sum(1), out=np.zeros(10), where=matrix.sum(1) != 0).tolist(),
                                  "confusion_matrix": matrix.tolist()}
        peak[0] = max(peak[0], process.memory_info().rss)
        return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                "iterations": iterations, "warmup": 10, "peak_rss_bytes": peak[0],
                "provider": "CPUExecutionProvider", "threads": 4, "batch_size": 1,
                "latency_scope": "session.run only; preprocessing excluded", **evaluations}
    finally:
        stop.set()
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    atomic_json(args.output, benchmark(args.model, args.manifest, args.iterations))

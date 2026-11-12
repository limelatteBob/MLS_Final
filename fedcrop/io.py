"""Atomic artifacts and deliberately narrow public telemetry."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


PUBLIC_FIELDS = {
    "event", "round", "client_id", "status", "epsilon", "epsilon_cap", "steps",
    "sample_rate", "noise_multiplier", "selected", "survivors", "model_version",
    "macro_f1", "accuracy", "simulated_seconds", "wall_seconds", "wire_bytes",
    "update_bytes", "secagg_overhead_bytes", "coverage", "abort_rate", "reason",
    "battery", "profile", "assigned_model", "download_bytes", "run_id", "config_hash",
}


def emit(path, **fields):
    unknown = set(fields) - PUBLIC_FIELDS
    if unknown:
        raise ValueError(f"Telemetry field is not approved: {sorted(unknown)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), **fields},
                                allow_nan=False) + "\n")


def host_info():
    import psutil
    import torch
    return {"platform": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total, "torch": torch.__version__,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@contextmanager
def run_lock(directory):
    """Kernel-held single-writer lock, automatically released after a crash."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / ".run.lock").open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError("Another process is writing this run directory") from exc
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

"""Run with .venv_cpu Python: import the full client stack before measuring RSS."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import fedcrop.federation
import fedcrop.experiment
from fedcrop.resources import profile_host
if __name__ == "__main__":
    if torch.version.cuda is not None:
        raise RuntimeError("Expected a CPU-only PyTorch installation")
    profile_host("artifacts/profile_cpu_head.json", repeats=5, freeze_features=True, normalize=True)

"""Fixed configuration, three FL seeds conditional on one public initialization."""
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedcrop.experiment import load_config, run
from fedcrop.io import atomic_json, read_json
from fedcrop.warmup import evaluate_run


if __name__ == "__main__":
    logging.getLogger("flwr").setLevel(logging.WARNING)
    logging.getLogger("flwr").propagate = False
    experiments = []
    for seed in [42, 43, 44]:
        experiments.append(load_config("configs/improved.json", {"seed": seed, "output": f"runs/improved_seed{seed}"}))
    experiments.append(load_config("configs/improved.json", {"output": "runs/improved_nonprivate",
                        "dp": False, "secure_aggregation": False}))
    atomic_json("artifacts/improved_study_protocol.json", {
        "experiments": experiments, "selection": "fixed learning rate .001; public model selected solely on validation",
        "scope": "Three federated seeds conditional on a shared seed42 public initialization; not three independent end-to-end seeds",
        "test_policy": "All training and validation results complete before final held-out evaluation"})
    for config in experiments:
        output = Path(config["output"])
        if not (output / "summary.json").exists():
            run(config, resume=(output / "state.pt").exists())
    for config in experiments:
        evaluate_run(config["output"])
    print("Completed fixed study and final evaluations", flush=True)

"""Conservative whole-process peak including six simulated clients and SecAgg+."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fedcrop.io import read_json, atomic_json
from fedcrop.data import ImageDataset, partition
from fedcrop.model import CropCNN, get_weights
from fedcrop.privacy import PrivacyLedger
from fedcrop.training import train_local
from fedcrop.federation import secure_round
from fedcrop.resources import PeakRSS


if __name__ == "__main__":
    assert torch.version.cuda is None
    torch.set_num_threads(4)
    manifest = read_json("data/manifest_v2.json")
    rows = [r for r in manifest["rows"] if r["split"] == "train"]
    parts = partition(rows)
    datasets = [ImageDataset([rows[i] for i in ids], augment=True) for ids in parts]
    config = {"device": "cpu", "normalize": True, "freeze_features": True,
              "learning_rate": .001, "physical_batch_size": 8, "clip_norm": 1.}
    with TemporaryDirectory(dir="artifacts") as folder:
        ledgers = [PrivacyLedger(Path(folder) / f"{i}.json", "probe", i, 8, 1e-5, create=True) for i in range(6)]
        with PeakRSS() as memory:
            initial = get_weights(CropCNN(normalize=True))
            def train(client, weights):
                return train_local(weights, datasets[client], config, seed=client, steps=2,
                                   batch_size=16, ledger=ledgers[client], sigma=5.)[0]
            aggregate, protocol = secure_round(initial, list(range(6)), train, 0)
        atomic_json("artifacts/cpu_round_probe.json", {"torch": torch.__version__,
                    "peak_rss_bytes": memory.peak, "secure_round_completed": aggregate is not None,
                    "wire_bytes": protocol["wire_bytes"], "logical_batch_size": 16,
                    "physical_batch_size": 8, "steps_per_client": 2,
                    "scope": "Whole host process with six simulated clients, image loading, accountants and SecAgg+; no edge hardware claim"})

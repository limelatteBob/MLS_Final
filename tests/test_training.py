import torch
from torch.utils.data import TensorDataset

from fedcrop.training import train_local
from fedcrop.model import CropCNN, get_weights
from fedcrop.privacy import PrivacyLedger


def test_real_dp_steps_and_recovery(tmp_path):
    torch.set_num_threads(2)
    dataset = TensorDataset(torch.rand(8, 3, 96, 96), torch.arange(8) % 10)
    config = {"device": "cpu", "learning_rate": .03, "clip_norm": 1., "physical_batch_size": 2}
    ledger = PrivacyLedger(tmp_path / "ledger.json", "test", 0, 8, 1e-5, create=True)
    weights, stats = train_local(get_weights(CropCNN()), dataset, config, seed=1,
                                steps=2, batch_size=4, ledger=ledger, sigma=3.)
    assert stats["steps"] == 2 and ledger.entries == 2
    before = ledger.epsilon
    resumed = PrivacyLedger(tmp_path / "ledger.json", "test", 0, 8, 1e-5)
    train_local(weights, dataset, config, seed=2, steps=1, batch_size=4, ledger=resumed, sigma=3.)
    assert resumed.entries == 3 and resumed.epsilon > before

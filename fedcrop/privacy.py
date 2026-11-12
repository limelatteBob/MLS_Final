"""Conservative write-ahead record-level RDP accounting.

Reservations are persisted BEFORE a logical DP step. A crash can overcount a step,
but cannot recover spent privacy by restoring a model checkpoint.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis import rdp

from .io import atomic_json, read_json


class BudgetExhausted(RuntimeError):
    pass


@lru_cache(maxsize=256)
def _one_step_rdp(sigma, rate):
    return rdp.compute_rdp(q=rate, noise_multiplier=sigma, steps=1, orders=RDPAccountant.DEFAULT_ALPHAS)


def epsilon_for_history(history, delta):
    if not history:
        return 0.0
    composed = sum(_one_step_rdp(sigma, rate) * count for sigma, rate, count in history)
    epsilon, _ = rdp.get_privacy_spent(orders=RDPAccountant.DEFAULT_ALPHAS, rdp=composed, delta=delta)
    return float(epsilon)


class PrivacyLedger:
    def __init__(self, path, run_id, client_id, epsilon_cap, delta, *, create=False):
        self.path = Path(path)
        self.run_id, self.client_id = run_id, client_id
        self.cap, self.delta = epsilon_cap, delta
        self.accountant = RDPAccountant()
        self.entries = 0
        if self.path.exists():
            record = read_json(self.path)
            expected = (run_id, client_id, epsilon_cap, delta)
            actual = tuple(record[k] for k in ["run_id", "client_id", "epsilon_cap", "delta"])
            if expected != actual:
                raise ValueError("Privacy ledger identity/config mismatch")
            self.accountant.history = [tuple(x) for x in record["history"]]
            self.entries = record["reserved_steps"]
            if sum(x[2] for x in self.accountant.history) != self.entries:
                raise ValueError("Corrupt privacy ledger")
        elif create:
            self.save()
        else:
            raise FileNotFoundError("Missing privacy ledger: refusing to reset spent budget")

    @property
    def epsilon(self):
        return epsilon_for_history(self.accountant.history, self.delta)

    def can_step(self, sigma, rate):
        if sigma <= 0 or not 0 < rate <= 1:
            raise ValueError("Invalid DP noise or sampling rate")
        candidate = copy.deepcopy(self.accountant)
        candidate.step(noise_multiplier=sigma, sample_rate=rate)
        return epsilon_for_history(candidate.history, self.delta) <= self.cap

    def reserve(self, sigma, rate):
        if not self.can_step(sigma, rate):
            raise BudgetExhausted("Next logical step would exceed epsilon cap")
        self.accountant.step(noise_multiplier=sigma, sample_rate=rate)
        self.entries += 1
        self.save()

    def save(self):
        atomic_json(self.path, {"schema": 1, "run_id": self.run_id, "client_id": self.client_id,
                               "epsilon_cap": self.cap, "delta": self.delta,
                               "epsilon": self.epsilon, "reserved_steps": self.entries,
                               "history": self.accountant.history})


def calibrate_noise(cap, delta, rate, steps):
    return float(get_noise_multiplier(target_epsilon=cap * 0.98, target_delta=delta,
                                     sample_rate=rate, steps=steps, accountant="rdp"))

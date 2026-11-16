"""Time budget and per-run memoization for the corrected MCTS experiment."""

import json
import math
import os
import time
from pathlib import Path

from rdkit import Chem


class SearchBudgetExceeded(RuntimeError):
    pass


def check_budget():
    deadline = os.environ.get("GOBLET_SEARCH_DEADLINE_EPOCH")
    if deadline and time.time() >= float(deadline):
        raise SearchBudgetExceeded("search wall-time budget reached")


def install_request_timeout(seconds=600):
    """Bound existing reward modules' serial HTTP calls in this process only."""
    import requests

    original_post = requests.post

    def bounded_post(*args, **kwargs):
        check_budget()
        requested = kwargs.get("timeout", seconds)
        if isinstance(requested, (int, float)):
            kwargs["timeout"] = (10, min(float(requested), seconds))
        elif requested is None:
            kwargs["timeout"] = (10, seconds)
        else:
            kwargs["timeout"] = (min(requested[0] or 10, 10),
                                 min(requested[1] or seconds, seconds))
        return original_post(*args, **kwargs)

    requests.post = bounded_post


class BudgetedReward:
    """Cache successful deterministic scores; retain a journal of every call.

    Existing target-specific reward formulas and pocket constraints are delegated
    unchanged. Failure sentinels are not cached. There is no cross-run preloading.
    """

    def __init__(self, delegate, journal=None):
        self.delegate = delegate
        self.cache = {}
        self.calls = 0
        self.cache_hits = 0
        self.journal = Path(journal) if journal else None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def reward(self, mol):
        check_budget()
        canonical = Chem.Mol(mol)
        for atom in canonical.GetAtoms():
            atom.SetAtomMapNum(0)
        key = Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=True)
        cached = key in self.cache
        started = time.monotonic()
        value = self.cache[key] if cached else float(self.delegate.reward(mol))
        if not math.isfinite(value):
            raise ValueError(f"non-finite reward for {key}: {value}")
        if value > self.vmin:
            self.cache[key] = value
        self.calls += 1
        self.cache_hits += int(cached)
        if self.journal:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"SMILES": key, "Reward": value,
                    "cache_hit": cached, "elapsed_s": time.monotonic()-started,
                    "finished_at_epoch": time.time()}, allow_nan=False) + "\n")
        return value

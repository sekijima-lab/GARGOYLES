"""Exercise the standard CLI with pretrained sampling and QED on CPU."""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch

# Set before importing torch or the generator so this check uses CPU only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["GARGOYLES_RANDOM_SEED"] = "42"
os.environ.pop("GOBLET_SEARCH_DEADLINE_EPOCH", None)
os.environ.pop("GOBLET_REWARD_JOURNAL", None)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import pandas as pd
    import torch

    original_load = torch.load

    def load_cpu(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    with tempfile.TemporaryDirectory(prefix="gargoyles-mcts-smoke-") as directory:
        output_dir = Path(directory)
        output = output_dir / "qed.csv"
        arguments = [str(ROOT / "Generator/mcts.py"), "mcts.reward_name=QED",
                     "mcts.init_smiles=c1ccccc1", "mcts.n_step=3", "mcts.n_step_fine=2",
                     f"mcts.output_fname={output}",
                     f"hydra.run.dir={output_dir / 'hydra'}"]
        with patch.object(sys, "argv", arguments), patch.object(torch, "load", side_effect=load_cpu):
            runpy.run_path(str(ROOT / "Generator/mcts.py"), run_name="__main__")
        status = json.loads(output.with_suffix(".status.json").read_text())
        frame = pd.read_csv(output)
        assert status["complete"] and status["completed_steps"] == 3, status
        assert status["stop_reason"] == "step_budget", status
        assert len(frame) > 0 and frame["FineStepComplete"].all(), frame
        assert {"SMILES", "Reward", "Sim", "Step", "Imp", "FineStepComplete"} <= set(frame.columns)
        assert frame["Step"].max() <= 3
        print(f"CPU smoke passed: {len(frame)} output rows, 3 completed Fine steps")


if __name__ == "__main__":
    main()

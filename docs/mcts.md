# Corrected MCTS

`Generator/mcts.py` is the standard single-objective search entry point. It
incorporates the corrections previously evaluated in `mcts_fixed.py`.

## Search behavior

- Coarse simulation passes the best raw reward among locally improving
  candidates to backup; `g(R) = R / (1 + abs(R))` is applied once.
- Fine backup propagates its already normalized rollout value directly.
  Invalid-rollout penalties are bounded at -1 for ancestor updates.
- Candidate selection retains the highest reward and up to two random distinct
  SMILES. Candidate rows remain DataFrames, avoiding malformed NaN nodes.
- Subtree attachment updates descendant depths. Depth-limited branches can be
  removed, and the first observed negative backup is retained.
- `mcts.n_step` counts completed Fine iterations, including a final partial
  Coarse batch when the requested total is not divisible by `mcts.n_step_fine`.
  The existing local improvement gate and fragment-removal semantics remain.

## Running

Use the existing GARGOYLES environment and pretrained checkpoints. For example:

```bash
python Generator/mcts.py mcts.reward_name=QED mcts.init_smiles=c1ccccc1 \
  mcts.n_step=100 mcts.n_step_fine=20 mcts.output_fname=qed.csv
python test/test_mcts.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python test/smoke_mcts_cpu.py
```

The entry point uses `mcts.n_step` (default 1000), `mcts.n_step_fine` (default
20; must be positive), and `mcts.init_smiles` (default benzene). The previous entry point hardcoded
2000 total iterations, 100 iterations per Coarse simulation, and a propane seed.
Reward formulas and the existing reward factory are unchanged by this update.
Boltz extensions developed on `boltzina` are outside this change.

## Budgets, caching, and outputs

- `GARGOYLES_RANDOM_SEED`: optional Python, NumPy, and PyTorch seed.
- `GOBLET_SEARCH_DEADLINE_EPOCH`: optional Unix timestamp checked during search
  and before reward evaluation, including cache hits. Without it, no search
  deadline is imposed. This is cooperative stopping, not a hard process kill;
  an in-progress operation may finish after the deadline.
- `GOBLET_REWARD_JOURNAL`: optional JSONL path recording finite returned reward
  values, failure sentinels, cache hits, and timings. Calls that raise are not
  journaled. Each run caches successful deterministic rewards using canonical
  isomeric SMILES without atom maps; values at or below `vmin` are retried.
- HTTP POST calls made through `requests.post` receive a maximum 10-second
  connection timeout and 600-second read timeout in this process. This does not
  impose a timeout on subprocess-based rewards.

CSV output is written beneath `data/result/` (or to an absolute
`mcts.output_fname`). It includes `SMILES`, `Reward`, `Step`, `Imp`, `Sim`, and
`FineStepComplete`. Observations evaluated during an interrupted Fine iteration
are retained with `FineStepComplete=False`.

After each Coarse simulation and when search exits, the CSV and accompanying
`.status.json` are each replaced atomically. Status includes requested/completed
iterations, stopping reason, reward/cache counts, and Coarse backup records.
A time-limited run is not equivalent to a completed iteration budget; inspect
`complete`, `completed_steps`, and `stop_reason` before comparing results.

## Published paper version

The pre-update published-paper code is preserved at the
[`paper-version` tag](https://github.com/sekijima-lab/GARGOYLES/tree/paper-version).
Search trajectories and outputs can change with the corrected implementation.

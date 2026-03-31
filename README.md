# Online RL Alignment (GS-only Rebuttal Version)

This repository contains a focused implementation of GS-based experiments for online RL alignment in a generalized bilinear preference model (GBPM) setting.


## What is implemented

- GS algorithm implementation in `algo/gs.py` (`gs_s2p_cvxpy`).
- Environment and link functions in `env.py`.
- Regret/evaluation utilities in `regret.py`.
- Plotting and fit-metric utilities in `plot.py`.
- GS experiment runner script in `tests/test_gs.py`.

## Repository layout

- `algo/gs.py`: core GS algorithm.
- `tests/test_gs.py`: main script used to run GS experiments.
- `env.py`: GBPM environment and reward/link definitions.
- `regret.py`: regret computation and result serialization helpers.
- `plot.py`: plotting and curve-fit analysis helpers.
- `utils.py`: small utility functions (timestamps, progress bar, etc.).

## Environment setup

For the setup refer to [setup_env.sh](setup_env.sh).

## How to run

Run the GS experiment script:

```bash
python tests/test_gs.py
```

The script parameters (horizon, seeds, eta sweep, estimator settings, etc.) are defined at the top of `tests/test_gs.py`.

## Experiment settings used in rebuttal figures

The rebuttal experiments were run with GS in GBPM using the settings below (configured in `tests/test_gs.py`):

- Link/model: `mu_name = "logistic"`
- Feature mode: `phi_mode = "random"` (uniform/random spherical features)
- Regularizer: `reg_type = "reverse_kl"`
- Horizon: `T = 10000`
- Rank: `r = 1`
- Eta sweep: `eta in [10^-2, ..., 10^3]` (log-spaced; corresponds to the `-2` to `3` range in figure names)

Two main problem-size settings were used for the plots:

- `d = 5`, `K = 20`, `S = 5`
- `d = 10`, `K = 40`, `S = 10`


## Outputs

Results are written under:

- `results/test_gs/<run_name>/`

Typical outputs include:

- per-seed `.npz` files
- summary `.npz` files
- generated plots (regret and fit diagnostics)
- `eta_summary.json`
- run manifest `.jsonl`

## Results

See here [results.pdf](./results.pdf)

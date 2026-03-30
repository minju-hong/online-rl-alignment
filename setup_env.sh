#!/usr/bin/env bash
set -euo pipefail

# 1) Go to project
cd /home/junghyun/online_rlhf_gbpm

# 2) Create uv virtual environment (Python 3.11)
uv venv .venv --python 3.11

# 3) Activate it
source .venv/bin/activate

# 4) Install dependencies used by this codebase
uv pip install --upgrade pip setuptools wheel
uv pip install numpy scipy matplotlib cvxpy

# 5) Run one of your scripts (example: GS test)
python tests/test_gs.py
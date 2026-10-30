#!/usr/bin/env bash
# Reproduce every number and figure in the README. Sequential on purpose
# (single process, < ~15 min on a laptop, < 300 MB RAM).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../.venv/bin/python}
$PY fetch_data.py
for s in 01_density_checks 02_recovery_exgauss 03_recovery_ddm 04_model_recovery 05_spectral_validation 06_hyperaktiv 07_flanker_ddm verify_dois; do
  echo "== $s"; $PY -u "$s.py"
done

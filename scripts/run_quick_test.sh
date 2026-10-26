#!/usr/bin/env bash
# Smoke test: reduced configuration on one dataset (results are NOT the paper's).
set -e
cd "$(dirname "$0")/.."
python -m xqmids --datasets UNSW-NB15 --quick --output outputs_quick

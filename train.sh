#!/bin/bash
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python3 -m internet_speak.train \
    --data_dir=data
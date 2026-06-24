#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

DATA_DIR=${DATA_DIR:-data/symphomotion-dataset}
CSV_PATH=${CSV_PATH:-data/symphomotion.csv}

mkdir -p "$DATA_DIR"
modelscope download --dataset Grenoble/symphomotion-dataset --local_dir "$DATA_DIR"
scripts/prepare_csv.py --root "$DATA_DIR" --output "$CSV_PATH"

echo "Dataset downloaded to: $DATA_DIR"
echo "CSV manifest written to: $CSV_PATH"

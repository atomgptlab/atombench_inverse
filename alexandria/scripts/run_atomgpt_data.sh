#!/bin/bash
set -e
mkdir -p agpt_alexandria
uv pip install jarvis-tools==2024.4.* pymatgen numpy pandas tqdm
export DEBUG="true"
python scripts/id_prop.py \
    --csv-files dataset1.csv dataset2.csv \
    --id-key mat_id \
    --target Tc \
    --output agpt_alexandria \
    --seed 123 \
    --max-size 1000




#!/bin/bash
set -e
mkdir -p agpt_tc_supercon
uv pip install jarvis-tools pymatgen numpy pandas tqdm
python scripts/data_preprocess.py atomgpt  --dataset dft_3d --output ./atomgpt_data  \
                                           --target Tc_supercon --max-size 1000 --seed 123

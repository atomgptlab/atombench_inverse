#!/bin/bash
set -e
mkdir -p atomgpt_data
uv pip install jarvis-tools pymatgen numpy pandas tqdm
python scripts/data_preprocess.py atomgpt  --dataset dft_3d --output ./atomgpt_data  \
                                           --target Tc_supercon --seed 123 --max-size 20
rm -rf JVASP*

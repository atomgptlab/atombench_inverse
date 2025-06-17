#!/bin/bash
set -e
uv pip install jarvis-tools==2024.4.* pymatgen numpy pandas tqdm
export DEBUG="true"
python scripts/alexandria_preprocess.py atomgpt \
       --csv-files dataset1.csv dataset2.csv \
       --output ./atomgpt_data --max-size 1000 --seed 123
rm agm*

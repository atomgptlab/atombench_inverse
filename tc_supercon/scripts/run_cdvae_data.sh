#!/bin/bash
set -e
mkdir -p ../models/cdvae/data/supercon/
uv pip install jarvis-tools pymatgen numpy pandas tqdm
python scripts/data_preprocess.py cdvae    --dataset dft_3d --output .    \
                                           --target Tc_supercon --seed 123 --max-size 20
python - <<'PYCODE'
import os
path = "../models/cdvae/data/supercon"
files = ["train.csv", "test.csv", "val.csv"]
for file in files:
	file_path = os.path.join(path,file)
	if os.path.exists(file_path):
		os.remove(file_path)
	os.rename(file, file_path)
yaml_path = "../models/cdvae/conf/data/supercon.yaml"
PYCODE


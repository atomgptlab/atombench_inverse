#!/bin/bash
# run data preprocessor
python scripts/data_preprocess.py flowmm --dataset dft_3d --output .  \
                                         --target Tc_supercon --max-size 1000 --seed 123

# move everything to the right spot
mkdir -p ../models/flowmm/data/supercon
python - <<'PYCODE'
import os
path = "../models/flowmm/data/supercon"
files = ["train.csv", "test.csv", "val.csv"]
for file in files:
	file_path = os.path.join(path,file)
	if os.path.exists(file_path):
		os.remove(file_path)
	os.rename(file, file_path)
PYCODE





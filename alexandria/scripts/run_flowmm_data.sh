#!/bin/bash
# run data preprocessor
python scripts/alexandria_preprocess.py flowmm \
       --csv-files dataset1.csv dataset2.csv \
       --output . --seed 123 --max-size 20

# move everything to the right spot
mkdir -p ../models/flowmm/data/alexandria
python - <<'PYCODE'
import os
path = "../models/flowmm/data/alexandria"
files = ["train.csv", "test.csv", "val.csv"]
for file in files:
	file_path = os.path.join(path,file)
	if os.path.exists(file_path):
		os.remove(file_path)
	os.rename(file, file_path)
PYCODE







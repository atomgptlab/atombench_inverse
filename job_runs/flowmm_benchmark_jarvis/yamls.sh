#!/bin/bash

eval "$(conda shell.bash hook)"

# remove yamls that might exist from past pipeline runs
rm -f models/flowmm/src/flowmm/rfm/manifolds/stats_supercon*
rm -f models/flowmm/src/flowmm/rfm/manifolds/stats_alex*
python - <<'PYCODE'
import os
files = ["atom_density.yaml", "spd_pLTL_stats.yaml", "spd_std_coef.yaml", "lattice_params_stats.yaml"]
path = "models/flowmm/src/flowmm/rfm/manifolds"
for file in files:
        filepath = os.path.join(path,file)
        if os.path.exists(filepath):
                os.remove(filepath)
PYCODE

# create necessary yamls for training and inference
cd models/flowmm
python -u -m flowmm.rfm.manifolds.spd
python -u -m flowmm.rfm.manifolds.lattice_params
python -u -m flowmm.model.standardize data=supercon
python -u -m flowmm.model.standardize data=alexandria

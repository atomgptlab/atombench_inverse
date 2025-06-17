#!/bin/bash
# remove yamls that might exist from past pipeline runs
rm -f /lab/mml/kipp/677/jarvis/rhys/benchmarks/models/flowmm/src/flowmm/rfm/manifolds/stats_supercon*
rm -f /lab/mml/kipp/677/jarvis/rhys/benchmarks/models/flowmm/src/flowmm/rfm/manifolds/stats_alex*
python - <<'PYCODE'
import os
files = ["atom_density.yaml", "spd_pLTL_stats.yaml", "spd_std_coef.yaml", "lattice_params_stats.yaml"]
path = "/lab/mml/kipp/677/jarvis/rhys/benchmarks/models/flowmm/src/flowmm/rfm/manifolds"
for file in files:
        filepath = os.path.join(path,file)
        if os.path.exists(filepath):
                os.remove(filepath)
PYCODE

# create necessary yamls for training and inference
cd /lab/mml/kipp/677/jarvis/rhys/benchmarks/models/flowmm
python -u -m flowmm.rfm.manifolds.spd
python -u -m flowmm.rfm.manifolds.lattice_params
python -u -m flowmm.model.standardize data=supercon
python -u -m flowmm.model.standardize data=alexandria

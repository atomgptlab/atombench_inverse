#!/bin/bash

#SBATCH --job-name=flow_yamls
#SBATCH --output=/lab/mml/kipp/677/jarvis/rhys/benchmarks/job_runs/flowmm_benchmark_jarvis/fl_studio.out
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4                # Run 4 tasks (processes) on the node
#SBATCH --cpus-per-task=2                  # 2 CPU cores per task (for multi‐threaded code)
#SBATCH --time=24:00:00                    # Max walltime (HH:MM:SS)
#SBATCH --mem=64G                          # Total RAM for the job (8 GB)

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

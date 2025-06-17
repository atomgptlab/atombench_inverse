#!/bin/bash
conda activate flowmm

python inspect_pt.py \
   --pt_path /lab/mml/kipp/677/jarvis/rhys/benchmarks/job_runs/flowmm_benchmark_test/outputs/rfmcsp-conditional-supercon/cxkrfmgi/checkpoints/inferences/consolidated_reconstruct.pt \
   --output_csv AI-AtomGen-prop-dft_3d-test-rmse.csv

mv ../../models/flowmm/AI-AtomGen-prop-dft_3d-test-rmse.csv .

python plot_error_distribution.py

#!/bin/bash
conda activate flowmm

rm -f job_runs/flowmm_benchmark_jarvis/AI-AtomGen*

python inspect_pt.py \
   --pt_path job_runs/flowmm_benchmark_jarvis/outputs/rfmcsp-conditional-*/????????/checkpoints/inferences/consolidated_reconstruct.pt \
   --output_csv AI-AtomGen-prop-dft_3d-test-rmse.csv

mv models/flowmm/AI-AtomGen-prop-dft_3d-test-rmse.csv .


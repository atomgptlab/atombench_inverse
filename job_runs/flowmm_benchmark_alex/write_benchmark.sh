#!/bin/bash
conda activate flowmm

rm -f job_runs/flowmm_benchmark_alex/AI-AtomGen*

python inspect_pt.py \
   --pt_path job_runs/flowmm_benchmark_alex/outputs/rfmcsp-conditional-*/????????/checkpoints/inferences/consolidated_reconstruct.pt \
   --output_csv job_runs/flowmm_benchmark_alex/AI-AtomGen-prop-dft_3d-test-rmse.csv

mv models/flowmm/AI-AtomGen-prop-dft_3d-test-rmse.csv .


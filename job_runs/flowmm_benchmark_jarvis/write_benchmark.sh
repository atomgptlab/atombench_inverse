#!/bin/bash
set -e
eval "$(conda shell.bash hook)"
conda activate flowmm
source scripts/absolute_path.sh
ROOT="${ABS_PATH%/}"

rm -f "${ROOT}/job_runs/flowmm_benchmark_jarvis/AI-AtomGen*"

FLOW_ROOT="${ROOT}/job_runs/flowmm_benchmark_jarvis/outputs"
pt_path="$(find "${FLOW_ROOT}" -type f -path '*/checkpoints/inferences/consolidated_reconstruct.pt' 2>/dev/null | xargs -r ls -1 -t 2>/dev/null | head -n1)"
if [[ -z "${pt_path}" ]]; then
  echo "ERROR: consolidated_reconstruct.pt not found under ${FLOW_ROOT}"
  exit 1
fi

out_csv="${ROOT}/job_runs/flowmm_benchmark_jarvis/AI-AtomGen-prop-dft_3d-test-rmse.csv"

python "${ROOT}/job_runs/flowmm_benchmark_jarvis/inspect_pt.py" \
   --pt_path "${pt_path}" \
   --output_csv "${out_csv}" \
   --test_csv "${ROOT}/models/flowmm/data/supercon/test.csv" \
   --dump_json true

# keep it in flowmm_benchmark_jarvis; only move fallback
if [[ -f "${out_csv}" ]]; then
  echo "Wrote ${out_csv}"
elif [[ -f "${ROOT}/models/flowmm/AI-AtomGen-prop-dft_3d-test-rmse.csv" ]]; then
  mv "${ROOT}/models/flowmm/AI-AtomGen-prop-dft_3d-test-rmse.csv" "${ROOT}/job_runs/flowmm_benchmark_jarvis/"
  echo "Moved fallback CSV to ${ROOT}/job_runs/flowmm_benchmark_jarvis/"
else
  echo "WARN: CSV not found where expected."
fi


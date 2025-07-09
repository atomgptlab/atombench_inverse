# benchmarks

#### Coming soon: CrystalGen software in github.com/atomgptlab that automates benchmark experiments and crystal structure generation!

## Note:
This repository is still undergoing development. Currently, absolute paths are mostly used for sbatch scripts and other files. Soon, these will be
updated to evironment-variable-based paths to allow for easy reproducability.

 Near-term goals:
- Use environment-variable-paths instead of absolute paths
- Wrap each benchmark as a Snakmake workflow

## Compute benchmarks
1) git submodule update --init --recursive
2) conda install -n base -c conda-forge mamba
3) pip install uv dvc
4) cd tc_supercon; dvc repro -f
5) cd alexandria; dvc repro -f
6) cd job_runs/
7) Update 'wandb_api_key.sh' to contain a valid wandb_api_key
8) for dir in *jarvis/; sbatch "$dir"/conda_env.job; done
9) When the sbatch jobs from 8) are finished, execute the following command:
   for dir in */; do sbatch "$dir"/yamls*; done
10) When the sbatch jobs from 9) are finished, execute the following command:
   for dir in */; do sbatch "$dir"/train*; done
11) When the sbatch jobs from 10) are finished, update absolute paths and execute the following command:
   for dir in */; do sbatch "$dir"/train*; done
12) When the sbatch jobs from 11) are finished, update absolute paths and execute the following command:
   for dir in */; do sbatch "$dir"/inference*; done
13) When the sbatch jobs from 12) are finished, update absolute paths and execute the following command:
   for dir in */; do sbatch "$dir"/benchmark.job*; bash write_benchmark.sh*; done
14) When the sbatch jobs from 13) are finished, update absolute paths and execute the following command:
   bash loop.sh
15) When the sbatch jobs from 14) are finished, update absolute paths and execute the following command:
    python ../scripts/bar_chart.py

## Installation & Usage Tutorials
### AtomGPT
https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb

### CDVAE
https://colab.research.google.com/github/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/cdvae_example.ipynb#scrollTo=dMNVDRFTNM12

### FlowMM
https://colab.research.google.com/github/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/flowmm_example.ipynb#scrollTo=jl8B-XPE-GR3

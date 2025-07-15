# benchmarks

#### Coming soon: CrystalGen software in github.com/atomgptlab that automates benchmark experiments and machine learning crystal structure generation!

![Poster](poster.png)

## Compute benchmarks
1) Run 'git submodule update --init --recursive'
2) Run 'export CONDA_EXECUTABLE={path}' where '{path}' is the path to the 'conda.sh' executable on your machine
3) Run 'conda install -n base -c conda-forge mamba'
4) Run 'pip install uv dvc snakemake'
5) Populate 'scripts/wandb_api_key.sh' with a valid wandb_api_key
6) Run 'snakemake --cores all'

## Installation & Usage Tutorials
### AtomGPT
https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb

### CDVAE
https://colab.research.google.com/github/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/cdvae_example.ipynb#scrollTo=dMNVDRFTNM12

### FlowMM
https://colab.research.google.com/github/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/flowmm_example.ipynb#scrollTo=jl8B-XPE-GR3

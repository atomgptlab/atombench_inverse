# benchmarks

#### Coming soon: CrystalGen software in github.com/atomgptlab that automates benchmark experiments and machine learning crystal structure generation!

![Poster](poster.png)

## Install dependencies
```bash
git submodule update --init --recursive
conda install -n base -c conda-forge mamba
pip install uv dvc snakemake
```

## Compute benchmarks
First, populate 'scripts/wandb_api_key.sh' with a valid wandb_api_key

Then, run this command to automatically recompute benchmarks, metrics, and figures:
```bash
snakemake all --cores all
```

## Installation & Usage Tutorials
### AtomGPT
https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb

### CDVAE
https://colab.research.google.com/github/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/cdvae_example.ipynb#scrollTo=dMNVDRFTNM12

### FlowMM
https://colab.research.google.com/github/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/flowmm_example.ipynb#scrollTo=jl8B-XPE-GR3

# Atombench

The rapid development of generative AI models for materials discovery has created an urgent need for standardized benchmarks to evaluate their performance. In this work, we present $\textbf{AtomBench}$, a systematic benchmarking framework that comparatively evaluates three representative generative architectures-AtomGPT (transformer-based), CDVAE (diffusion variational autoencoder), and FlowMM (Riemannian flow matching)-for inverse crystal structure design. We train and evaluate these models on two high-quality DFT superconductivity datasets: JARVIS Supercon-3D and Alexandria DS-A/B, comprising over 9,000 structures with computed electron-phonon coupling properties.

## Install dependencies
1) Install models as submodules
2) Install mamba to speed up conda env creation
3) Install base python dependencies
```bash
git submodule update --init --recursive
conda install -n base -c conda-forge mamba
pip install uv dvc snakemake
```

## Compute benchmarks
1) Navigate to [scripts/absolute_path.sh](scripts/absolute_path.sh) and populate with the absolute path to this repository
2) Run this command to automatically recompute benchmarks, metrics, and figures:
```bash
snakemake all --cores all
```

## Installation & Usage Tutorials
### [AtomGPT](https://github.com/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)
### [CDVAE](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/cdvae_example.ipynb)
### [FlowMM](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/flowmm_example.ipynb)



# Atombench

The rapid development of generative AI models for materials discovery has created an urgent need for standardized benchmarks to evaluate their performance. In this work, we present $\textbf{AtomBench}$, a systematic benchmarking framework that comparatively evaluates three representative generative architectures-AtomGPT (transformer-based), CDVAE (diffusion variational autoencoder), and FlowMM (Riemannian flow matching)-for inverse crystal structure design. We train and evaluate these models on two high-quality DFT superconductivity datasets: JARVIS Supercon-3D and Alexandria DS-A/B, comprising over 9,000 structures with computed electron-phonon coupling properties.

## Installation Instructions
#### Step 1: Download this repository
Download this repository using the following command:
```bash
git clone git@github.com:atomgptlab/atombench_inverse.git
```
NOTE: This repository must be cloned using `ssh` rather than `https` due to constraints from the generative models used in this repository.


#### Step 2: Initialize the generative models
To recompute the benchmarks, the generative models need to be downloaded into the `models/` directory, and we automate this using `git submodules`. In the root directory of this repository, run the following command to download and initialize the generative models used in this study:
```bash
git submodule update --init --recursive
```

#### Step 3: Install `mamba`
Setting up the correct environments for some of the generative models can take quite a while, so we speed up this process by using `mamba` instead of `conda` for intensive tasks. Download `mamba` using the following command:
```bash
conda install -n base -c conda-forge mamba
```

#### Step 4: Create and activate a `conda` environment to host Atombench Python dependencies
Normally, it is best-practice to avoid installing Python packages to one's base `conda` environment. Make an environment to store required Python deps:
```bash
conda create --name atombench python=3.11 -y
conda activate atombench
```
NOTE: Becayse we are including `python=3.11` in the `conda create` command, we can use both `conda install` and `pip install` to install Python packages into the newly created `atombench` environment.

#### Step 5: Download Python dependencies
This repository recomputes the AtomBench benchmarks using a semi-automated `Snakemake` pipeline. For more information about `Snakemake`, visit their [documentation](https://snakemake.readthedocs.io/en/stable/) site. Moreover, we use `uv` to speed up downstream package installation, and we use `DVC` to automate dataset preprocessing.
```bash
pip install uv snakemake dvc
```

## Recompute Atombench Benchmarks
#### Step 1: Provide the location of the `atombench_inverse` repository
For this repository to execute its core functionalities, it must know its own location in the computer's filesystem. To accomplish this, locate a file in the `scripts/` directory called `absolute_path.sh`, and set the `ABS_PATH` environment variable equal to the repository's absolute path, e.g.
```bash
vi scripts/absolute_path.sh
```
then,
```bash
#!/bin/bash
export ABS_PATH="path/to/atombench_inverse/repository"
```
#### Step 2: Activate the `Snakemake` pipeline to recompute the benchmarks
As mentioned previously, we use an automated pipeline to compute these benchmarks. After the previous setup steps have been completed, run the pipeline using the following command:
```bash
snakemake all --cores all
```

## Generative Model Installation & Usage Tutorials
### [AtomGPT](https://github.com/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/atomgpt_example.ipynb)
### [CDVAE](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/cdvae_example.ipynb)
### [FlowMM](https://github.com/crhysc/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/flowmm_example.ipynb)



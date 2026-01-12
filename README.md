# Atombench

The rapid development of generative AI models for materials discovery has created an urgent need for standardized benchmarks to evaluate their performance. In this work, we present $\textbf{AtomBench}$, a systematic benchmarking framework that comparatively evaluates three representative generative architectures-AtomGPT (transformer-based), CDVAE (diffusion variational autoencoder), and FlowMM (Riemannian flow matching)-for inverse crystal structure design. We train and evaluate these models on two high-quality DFT superconductivity datasets: JARVIS Supercon-3D and Alexandria DS-A/B, comprising over 9,000 structures with computed electron-phonon coupling properties.

## Installation Instructions
#### Step 1: Confirm operating system, workload manager, and Python package manager
This repo is built for Linux-based High-Performance Computing (HPC) clusters that use the SLURM workload manager. It will not run on MacOS devices, Windows devices, and non-SLURM HPC clusters. A valid conda installation is also required, and it must have the capacity to be initialized using the Conda shell integration hook via the following command:
```bash
eval "$(conda shell.bash hook)"
```
Documentation about activating conda environments using this command can be found [here](https://docs.conda.io/projects/conda/en/latest/dev-guide/deep-dives/activation.html). To verify that using the shell hook works, running `conda deactivate` should remove the `(base)` prefix, and running `eval "$(conda shell.bash hook)"` should re-add the `(base)` prefix:
```bash
(base) [user@hpc-cluster ~]$ conda deactivate
[user@hpc-cluster atombench]$ eval "$(conda shell.bash hook)"
(base) [user@hpc-cluster ~]$
```

#### Step 2: Download this repository
Download this repository using the following command:
```bash
git clone git@github.com:atomgptlab/atombench_inverse.git
```
NOTE: This repository must be cloned using `ssh` rather than `https` due to constraints from the generative models used in this repository.


#### Step 3: Initialize the generative models
To recompute the benchmarks, the generative models need to be downloaded into the `models/` directory, and we automate this using `git submodules`. In the root directory of this repository, run the following command to download and initialize the generative models used in this study:
```bash
git submodule update --init --recursive
```


#### Step 4: Create and activate a `conda` environment to host Atombench Python dependencies
Normally, it is best-practice to avoid installing Python packages to one's base `conda` environment. Make an environment to store required Python deps:
```bash
conda create --name atombench python=3.11 -y
conda activate atombench
```
NOTE: Becayse we are including `python=3.11` in the `conda create` command, we can use both `conda install` and `pip install` to install Python packages into the newly created `atombench` environment.

#### Step 4: Download Python dependencies
This repository recomputes the AtomBench benchmarks using a semi-automated `Snakemake` pipeline. For more information about `Snakemake`, visit their [documentation](https://snakemake.readthedocs.io/en/stable/) site. Moreover, we use `uv` to speed up downstream package installation, and we use `DVC` to automate dataset preprocessing.
```bash
pip install uv snakemake dvc
```

## Recompute Atombench Benchmarks
#### Step 1: Provide the location of this atombench repository
For this repository to execute its core functionalities, it must know its own location in the computer's filesystem. To accomplish this, locate a file in the `scripts/` directory called `absolute_path.sh`, and set the `ABS_PATH` environment variable equal to the repository's absolute path, e.g.
```bash
vi scripts/absolute_path.sh
```
then,
```bash
#!/bin/bash
export ABS_PATH="path/to/this/repository"
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



#!/bin/bash

cd models/cdvae

conda env create -f env.yml -y
eval "$(conda shell.bash hook)"
conda activate cdvae
conda install -c conda-forge "torchmetrics<0.8" --yes
conda install mkl=2024.0 --yes
pip install "monty==2022.9.9"
conda install -c conda-forge "pymatgen>=2022.0.8,<2023" --yes
pip install pandas jarvis-tools
pip install --upgrade torch_geometric==1.7.0
pip install -e .

#!/bin/bash

cd models/flowmm

mamba env create -f environment.yml -y
eval "$(conda shell.bash hook)"
conda activate flowmm
pip install uv
uv pip install "jarvis-tools>=2024.5" "pymatgen>=2024.1" pandas numpy tqdm
uv pip install -e . \
	       -e remote/riemannian-fm \
	       -e remote/cdvae \
	       -e remote/DiffCSP-official
	
echo "Done"

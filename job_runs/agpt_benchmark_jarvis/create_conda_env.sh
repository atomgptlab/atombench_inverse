#!/bin/bash

cd models/atomgpt

conda create --name my_atomgpt python=3.10 -y
conda activate my_atomgpt

pip install uv
uv pip install -e .

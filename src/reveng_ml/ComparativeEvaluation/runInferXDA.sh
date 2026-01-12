#!/bin/bash

module load devel/miniforge/25.3.1-python-3.12
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate xda && python3 InferXDA.py $1 $2
conda deactivate
#!/bin/bash


cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate xda && python3 InferXDA.py $1 $2
conda deactivate
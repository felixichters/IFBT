#!/bin/bash

module load devel/miniforge/25.3.1-python-3.12
eval "$(conda shell.bash hook)"
conda create -n xda python=3.7 numpy scipy scikit-learn colorama
conda activate xda
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install pyelftools
conda deactivate
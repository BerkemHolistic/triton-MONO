#!/bin/bash

# Create the environment
conda create -y -n python310 python=3.10

# Activate it
source activate python310

# Install ipykernel
conda install -y ipykernel pip scipy scikit-learn

# Install the kernel and reload in the browser tab
python -m ipykernel install --name "python310" --user

pip install -r requirements.txt

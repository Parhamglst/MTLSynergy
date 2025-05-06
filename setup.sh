#!/bin/bash

# Create a conda environment with Python 3.13.2
conda create -y -n MTLSynergy python=3.13.2

# Activate the environment
conda activate MTLSynergy

# Install packages from requirements.txt
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi
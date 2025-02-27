#!/bin/bash
#SBATCH --account=def-mirzavan
#SBATCH --gpus-per-node=1
#SBATCH --mem=6000M
#SBATCH --time=0-12:00:00
python MTLSynergytrain.py
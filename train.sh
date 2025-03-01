#!/bin/bash
#SBATCH --account=def-mirzavan
#SBATCH --gpus-per-node=1
#SBATCH --mem=6000M
#SBATCH --time=0-12:00:00
module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
python MTLSynergytrain.py

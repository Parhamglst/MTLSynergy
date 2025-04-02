#!/bin/bash
#SBATCH --account=def-mirzavan
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=16000M
#SBATCH --time=0-72:00:00
module load python/3.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
python MTLSynergy2_train.py

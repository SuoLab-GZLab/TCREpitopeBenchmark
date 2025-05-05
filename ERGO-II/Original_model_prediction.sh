#!/bin/bash
#SBATCH -c 1
#SBATCH -o %j.out
#SBATCH -e %j.err


module load miniforge3
module load  cuda/11.0.3
source activate ERGOII
python prediciton_test.py
conda deactivate

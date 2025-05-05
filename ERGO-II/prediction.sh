#!/bin/bash
#SBATCH -c 1
#SBATCH -o %j.out
#SBATCH -e %j.err

source activate ERGOII
python Original_model_prediction.py
conda deactivate

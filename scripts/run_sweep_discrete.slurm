#!/bin/bash
#SBATCH --job-name=sweep_disc
#SBATCH --account=3263572
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate neural_odes

cd /home/3263572/NeuralODEs

wandb agent gaiagrossi-bocconi-university/NeuralODEs/cedblosn

conda deactivate
module unload /software/modules/miniconda3
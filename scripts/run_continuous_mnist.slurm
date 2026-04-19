#!/bin/bash
#SBATCH --job-name=ode_mnist
#SBATCH --account=3263572
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=60
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate neural_odes

cd /home/3263572/NeuralODEs

python -m scripts.train_continuous_mnist --epochs 30 --hidden_dim 124 --lr 1e-3

conda deactivate
module unload /software/modules/miniconda3
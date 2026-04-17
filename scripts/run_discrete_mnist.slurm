#!/bin/bash
#SBATCH --job-name=disc_mnist
#SBATCH --output=logs/disc_mnist_%j.out
#SBATCH --error=logs/disc_mnist_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neural-odes

cd ~/NeuralODEs

python scripts/train_discrete_mnist.py
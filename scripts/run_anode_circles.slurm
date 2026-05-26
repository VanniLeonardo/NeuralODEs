#!/bin/bash
#SBATCH --job-name=anode_circles
#SBATCH --account=3263572
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=240
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load /software/modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate neural_odes

cd /home/3263572/NeuralODEs

export WANDB_MODE=offline

python scripts/train_anode_circles.py \
  --epochs 200 \
  --n_samples 1000 \
  --batch_size 64 \
  --augment_dims 0,1,2,5 \
  --seeds 0,1,2 \
  --log_every 20 \
  --write_results

conda deactivate
module unload /software/modules/miniconda3

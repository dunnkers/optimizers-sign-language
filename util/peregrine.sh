#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=3000
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=dl-sign-language
#SBATCH --output=logs/slurm-%A_%a.out
# → do make sure /logs directory exists!

module load Python/3.8.2-GCCcore-9.3.0
source venv/bin/activate
pip3 install -r requirements.txt
python3 src/train_model_test.py

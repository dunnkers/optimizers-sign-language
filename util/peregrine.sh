#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=dl-sign-language
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --array=1
# → do make sure /logs directory exists!

module load Python/3.8.2-GCCcore-9.3.0
source venv/bin/activate
pip install -r requirements.txt
python src/train_model.py \
    --path /data/${USER}/data.csv \
    --name ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    --epochs ${SLURM_ARRAY_TASK_ID}

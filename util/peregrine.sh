#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=100000
#SBATCH --nodes=1
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --job-name=dl-sign-language
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --array=0-8
# → do make sure /logs directory exists!

module load TensorFlow
pip install -r requirements.txt --user
python src/train_model.py \
    --path /data/${USER}/data.csv \
    --name ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    --optimizer ${SLURM_ARRAY_TASK_ID} \
    --epochs 200 \
    --weights imagenet

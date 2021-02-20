#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=128000
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=dl-sign-language
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --array=0-7
# → do make sure /logs directory exists!

module load TensorFlow
source venv/bin/activate
pip install -r requirements.txt
python src/train_model.py \
    --path /data/${USER}/data.csv \
    --name ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    --optimizer ${SLURM_ARRAY_TASK_ID}
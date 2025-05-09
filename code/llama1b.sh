#!/bin/bash
#SBATCH --job-name=llama1b
#SBATCH --nodelist=dgx009
#SBATCH --partition=queue_dip_ingegneria
#SBATCH --gres=gpu:1
#SBATCH --mem=100G         
#SBATCH --error=Logs/llama1b.error.log  # Save errors with the job name
#SBATCH --output=Logs/llama1b.output.log  # Save standard output instead of discarding it

# Ensure Logs directory exists
mkdir -p Logs  

export TZ=Europe/Rome

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate fg


# Run the evaluation script
python3 /mnt/beegfs/home/fbetello/green_rag/Green-RAG/code/compute_dataset.py --llm llama1b --split test
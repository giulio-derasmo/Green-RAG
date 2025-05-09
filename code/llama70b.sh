#!/bin/bash
#SBATCH --job-name=llama70b
#SBATCH --nodelist=dgx008
#SBATCH --partition=queue_dip_ingegneria
#SBATCH --gres=gpu:3
#SBATCH --mem=300G         
#SBATCH --error=Logs/llama70b.error.log  # Save errors with the job name
#SBATCH --output=Logs/llama70b.output.log  # Save standard output instead of discarding it

# Ensure Logs directory exists
mkdir -p Logs  

export TZ=Europe/Rome

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate fg


# Run the evaluation script
python3 /mnt/beegfs/home/fbetello/green_rag/Green-RAG/code/compute_dataset.py --llm llama70b --split test --load_in_8bit 0
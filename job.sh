#!/bin/bash
#SBATCH --partition=singhlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00            # 
#SBATCH -J cjepa
#SBATCH --output=slurm_outputs/%A.out 

uv run main.py
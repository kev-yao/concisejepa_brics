#!/bin/bash
#SBATCH --partition=singhlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --time=12:00:00            # 
#SBATCH -J cjepa
#SBATCH --output=slurm_outputs/%A.out 

# python make_molfeat.py \
#   --all-smiles-csv /hpc/group/singhlab/user/me196/projects/moleculerep/runs/REVICE/data/moodeng/all_smiles.csv \
#   --output-path /hpc/group/singhlab/user/yk307/projects/concisejepa/data/DTI-datasets/moodeng/morgan_embeddings.pt

uv run main.py
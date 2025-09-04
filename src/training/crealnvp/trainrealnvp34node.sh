#!/bin/bash
#SBATCH -J RealNVP34
#SBATCH -p gpu_a100          # change to a listed partition if different
#SBATCH -t 99:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1             # one full A100 on gpu_a100; on gpu_mig this is one MIG
#SBATCH --mem=40G

# set -euo pipefail

# module purge
module load 2023r1 
module load cuda/11.6

source ~/venvs/ppf/bin/activate
# Run your program (use python, not a bare .py with srun)
srun python src/training/crealnvp/main_34.py

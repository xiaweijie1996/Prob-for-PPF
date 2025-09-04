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
module load 2023          # framework toolchain family (use the one you have)
module load Python/3.10.4-GCCcore-11.3.0  # example from SURF docs; adjust if needed

source ~/venvs/ppf/bin/activate
# Run your program (use python, not a bare .py with srun)
srun python src/training/crealnvp/main_34.py

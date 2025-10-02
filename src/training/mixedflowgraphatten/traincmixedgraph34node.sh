#!/bin/bash
#SBATCH -J CSMixedGraphAttention
#SBATCH -p gpu_a100           # use a GPU partition that 'sinfo' shows for your account
#SBATCH -t 99:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G

set -euo pipefail

module purge
module load 2023               # or 2023r1 if that’s the right family
module load CUDA/12.1.1        # pick the version you actually have
module load Python/3.11.3-GCCcore-12.3.0

# OPTION 1: Use site PyTorch module (if available)
# module load PyTorch/2.2.2-foss-2023b-CUDA-12.1.1

# OPTION 2: Use your own virtualenv (comment out if using site PyTorch)
source /gpfs/home4/wxia/Prob-for-PPF/.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Host: $(hostname)"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

srun python /gpfs/home4/wxia/Prob-for-PPF/src/training/mixedflowgraphatten/main_34.py

#!/bin/bash
#SBATCH --job-name=DDPM_CUDA
#SBATCH --account=project_465000568
#SBATCH --time=12:00:00
#SBATCH --output=DDPM_CUDA.txt
#SBATCH --error=DDPM_CUDA.txt
#SBATCH --exclusive
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=8

export PATH="/users/quistgaa/install_torch/bin:$PATH"

# Load modules
module load LUMI
module load singularity-bindings
module load aws-ofi-rccl

. ~/pt_rocm5.4.1_env/bin/activate

# Run the job
srun python3 ddpm_DANRA_conditional_wValid__128x128.py
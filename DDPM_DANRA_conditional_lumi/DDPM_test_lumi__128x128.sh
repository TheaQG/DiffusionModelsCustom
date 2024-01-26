#!/bin/bash
#SBATCH --job-name=DDPM_128x128_sdf
#SBATCH --account=project_465000568
#SBATCH --time=12:00:00
#SBATCH --output=DDPM_128x128_sdf_out.txt
#SBATCH --error=DDPM_128x128_sdf_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:8
#SBATCH --partition=small-g

export PATH="/users/quistgaa/install_torch/bin:$PATH"

# Load modules
module load LUMI

# Run the job
srun python3 ddpm_DANRA_conditional_wValid__128x128.py
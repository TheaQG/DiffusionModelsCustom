#!/bin/bash
#SBATCH --job-name=DDPM_64x64_lsmtopo_only
#SBATCH --account=project_465000568
#SBATCH --time=30:00:00
#SBATCH --output=DDPM_64x64_lsm_topo_only_out.txt
#SBATCH --error=DDPM_64x64_lsm_topo_only_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --partition=small-g

export PATH="/users/quistgaa/install_torch/bin:$PATH"

# Load modules
module load LUMI

# Run the job
srun python3 ddpm_DANRA_conditional__lsm_topo__wValid.py
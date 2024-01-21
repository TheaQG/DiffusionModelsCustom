#!/bin/bash
#SBATCH --job-name=zarr_file_creation
#SBATCH --account=project_465000568
#SBATCH --time=12:00:00
#SBATCH --output=zarr_file_creation_out.txt
#SBATCH --error=zarr_file_creation_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:1
#SBATCH --partition=small-g

export PATH="/users/quistgaa/install_torch/bin:$PATH"

# Load modules
module load LUMI

# Run the job
srun python3 daily_files_to_zarr.py
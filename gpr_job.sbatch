#!/usr/bin/bash
#SBATCH --job-name=gpr_run
#SBATCH --output=gpr_run.%j.out
#SBATCH --error=gpr_run.%j.err
#SBATCH --time=10:00:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=64GB
module load matlab
matlab -nodesktop < pred_team6.m
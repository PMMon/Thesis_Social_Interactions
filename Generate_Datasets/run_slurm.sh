#!/bin/bash
#SBATCH --job-name="SocialForceSimul"
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=all
#SBATCH --output=slurm/slurm-%j.out

eval "python  $@";


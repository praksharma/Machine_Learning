#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name GANS
#SBATCH --cpus-per-task 4
#SBATCH --ntasks=1
#SBATCH -o output.log
#SBATCH -e error.log
#SBATCH --gres=gpu:1
#SBATCH --account=scw1901
#SBATCH --partition=accel_ai

source /scratch/s.1915438/env/modulus/bin/activate

python code_wandb.py

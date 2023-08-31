#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name WANDB_trial
#SBATCH -o batch_output.log
#SBATCH -e batch_error.log
#SBATCH --gres=gpu:1
#SBATCH --account=scw1901
#SBATCH --partition=accel_ai


module load git/2.19.2 # mlflow needs git to play tennis

# python env
source /scratch/s.1915438/env/modulus/bin/activate

wandb agent prakhars962/parallel_try_2/6m61zudj

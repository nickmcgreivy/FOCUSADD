#!/bin/bash
#SBATCH --job-name=FilamentFOCUSADD       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=mcgreivy@princeton.edu

#module load cudatoolkit/10.1 cudnn/cuda-10.1/7.6.3 anaconda3/2019.10 rh/devtoolset/8
# User specific aliases and functions
#conda activate jax-env
module load anaconda3
conda activate jax-cpu

#module load anaconda3
#conda activate j
python filamentRun.py -n 10

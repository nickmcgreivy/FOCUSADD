#!/bin/bash
#SBATCH --job-name=BaselineFOCUSADD       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=mcgreivy@princeton.edu

module load anaconda3
conda activate j
cd 
cd FOCUSADD/focusadd
python focusadd_main.py -n 500 -nt 32 -nz 128 -ns 32 -nfc 5 -rc 2.0 -lr 0.001 -w 0.1 -o ../tests/validateWithFocus/validateFOCUS

#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=jupyter                                   # sets the job name
#SBATCH --output=notebook.out.%j                              # indicates a file to redirect STDOUT to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=notebook.out.%j                               # indicates a file to redirect STDERR to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=24:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=high                                           # set QOS, this will determine what resources can be requested
#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=1                                              # request 4 cpu cores be reserved for your node total
#SBATCH --gres=gpu:rtxa6000:3
#SBATCH --mem=128gb                                               # memory required by job; if unit is not specified MB will be assumed

srun python3 run_poem.py

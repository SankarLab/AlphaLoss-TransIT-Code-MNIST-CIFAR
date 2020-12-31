#!/bin/bash

#SBATCH -N 1
#SBATCH -n 2
##SBATCH -q wildfire                 # Run job under wildfire QOS queue
#SBATCH -t 0-10:00                  # wall time (D-HH:MM)
#SBATCH -o %j.out             # STDOUT (%j = JobId)
#SBATCH -e %j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when a job starts, stops, or fails
#SBATCH --mail-user=jcava@asu.edu # send-to address

source activate ~/.conda/envs/pytorch-1.30-gpu/
python mnist-noise-alpha.py $5 $1 $2 $3 $4 $6 $7> mnist-noise-p$5-alpha-$1.out
conda deactivate

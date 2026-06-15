#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --job-name=ood_eval

srun apptainer exec --nv --env-file .env container.sif /bin/bash run_eval.sh
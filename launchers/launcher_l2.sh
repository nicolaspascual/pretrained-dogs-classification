#!/bin/bash
#SBATCH --job-name="main transfer learning"
#SBATCH --workdir=.
#SBATCH --qos=training
#SBATCH --ntasks=4
#SBATCH --gres gpu:1
#SBATCH --time=12:00:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python l2norm_main.py
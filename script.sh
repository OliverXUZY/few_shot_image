#!/usr/bin/env bash
#
#SBATCH --output=./log/l_device_%j.out
#SBATCH --error=./log/e_device_%j.err
#SBATCH -J Mm  # give the job a name   
#***SBATCH --partition=batch_default ***
# 
# 1 node, 1 CPU per node (total 1 CPU), wall clock time of hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=4   ## Processors per node
#SBATCH --ntasks=1            ## Tasks
#SBATCH --gres=gpu:1          ## GPUs
#SBATCH --cpus-per-task=4     ## CPUs per task; number of threads of each task
#SBATCH -t 256:00:00          ## Walltime
#SBATCH --mem=40GB
#SBATCH -p lianglab
source ~/.bashrc


# python finetune.py --config=configs/clip/mini-imagenet/finetune_ViT.yaml
# python finetune.py --config=configs/clip/mini-imagenet/finetune_RN50.yaml
python test.py --config=configs/clip/mini-imagenet/test.yaml
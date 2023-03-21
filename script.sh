#!/usr/bin/env bash
#
#SBATCH --output=./log/l_device_%j.out
#SBATCH --error=./log/e_device_%j.err
#SBATCH -J test50m800M  # give the job a name   
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
#SBATCH -p research
source ~/.bashrc


echo "======== testing CUDA available ========"
python - << EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
EOF

echo "======== run with different inputs ========"

python finetune.py \
    --config=configs/clip/mini-imagenet/finetune_ViT.yaml \
    --n_batch_train $1 \
    --n_shot $2 \
    --sample_per_task $3

# python finetune.py --config=configs/clip/mini-imagenet/finetune_ViT_makeup_paper.yaml --n_shot 1 --sample_per_task 50 --path meta-mini-imagenet_ViT-B32_fs-centroid_5y1s_50m_800M
# python finetune.py --config=configs/clip/mini-imagenet/finetune_RN50.yaml
# python test.py --config=configs/clip/mini-imagenet/test.yaml 

# python finetune.py --config=configs/moco_v2/mini-imagenet/finetune.yaml


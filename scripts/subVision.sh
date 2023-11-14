#!/usr/bin/env bash
#
#SBATCH --output=./log/vision/l_multi_device_%j.out
#SBATCH --error=./log/vision/e_multi_device_%j.err
#SBATCH -J Mm  # give the job a name   
#***SBATCH --partition=batch_default ***
# 
# 1 node, 1 CPU per node (total 1 CPU), wall clock time of hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=1   ## Processors per node
#SBATCH --ntasks=1            ## Tasks
#SBATCH --gres=gpu:1          ## GPUs
#SBATCH --cpus-per-task=8     ## CPUs per task; number of threads of each task
#SBATCH -t 6:00:00          ## Walltime
#SBATCH --mem=40GB
#SBATCH -p lianglab
#SBATCH --exclude=euler[01-16],euler[20-28]
source ~/.bashrc
# 256:00:00          ## Walltime

echo "======== testing CUDA available ========"
echo "running on machine: " $(hostname -s)
python - << EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
EOF

echo "======== run with different inputs ========"

####################################################################################################################################################################################
####################################################################################################################################################################################
############################################# clip
##### mini-imagenet
# python finetune.py \
#     --config=configs/clip/mini-imagenet/finetune_ViT.yaml \
#     --n_batch_train $1 \
#     --n_shot $2 \
#     --sample_per_task $3

# python finetune.py --config=configs/clip/mini-imagenet/finetune_ViT.yaml
# python test.py --config=configs/clip/mini-imagenet/test.yaml \
#     --path "meta-mini-imagenet_ViT-B32_fs-centroid_15y4s_600m_800M"

# python finetune.py --config=configs/clip/mini-imagenet/finetune_ViT_lp.yaml 

# python finetune.py \
#     --config=configs/clip/mini-imagenet/finetune_ViT.yaml \
#     --n_batch_train 800 \
#     --n_shot 4 \
#     --sample_per_task 600 \
#     --path "meta-mini-imagenet_ViT-B32_fs-centroid_15y4s_600m_800M"

# continue to tune
# python finetune.py \
#     --config=configs/clip/mini-imagenet/finetune_ViT.yaml \
#     --path "meta-mini-imagenet_ViT-B32_fs-centroid_15y4s_600m_800M"

# python finetune.py --config=configs/moco_v2/mini-imagenet/finetune.yaml

# python test.py --config=configs/clip/mini-imagenet/test.yaml 

# python test.py \
#     --config=configs/clip/mini-imagenet/test.yaml \
#     --path "meta-mini-imagenet_ViT-B32_fs-centroid_15y10s_300m_200M" \
#     --exp vary_num_shot

##### tiered-imagenet


# python finetune.py --config=configs/clip/tiered-imagenet/finetune_ViT_stdFT.yaml


# python finetune.py \
#     --config=configs/clip/tiered-imagenet/finetune_ViT.yaml \
#     --n_batch_train 600 \
#     --n_shot 1 \
#     --sample_per_task 150

# continue to tune
# python finetune.py \
#     --config=configs/clip/tiered-imagenet/finetune_RN50.yaml \
#     --path "meta-tiered-imagenet_RN50_fs-centroid_15y4s_600m_800M"

python test.py --config=configs/clip/tiered-imagenet/test.yaml 

# python test.py \
#     --config=configs/clip/tiered-imagenet/test.yaml \
#     --path "meta-tiered-imagenet_ViT-B32_fs-centroid_15y1s_150m_200M" \
    # --exp vary_num_shot

####### domain-net
# python test.py --config=configs/clip/domain-net/test.yaml
# python finetune.py --config=configs/clip/domain-net/finetune_ViT.yaml
# python finetune.py --config=configs/clip/domain-net/finetune_RN50.yaml

####################################################################################################################################################################################
####################################################################################################################################################################################
############################################# dinov2
####### mini-imagenet
python finetune.py --config=configs/dinov2/mini-imagenet/finetune.yaml
# python test.py --config=configs/dinov2/mini-imagenet/test.yaml 

####### tiered-imagenet
# python finetune.py --config=configs/dinov2/tiered-imagenet/finetune.yaml
# python test.py --config=configs/dinov2/tiered-imagenet/test.yaml 

####### domain-net
# python test.py --config=configs/dinov2/domain-net/test.yaml
# python finetune.py --config=configs/dinov2/domain-net/finetune.yaml

####### CIFAR-FS
python test.py --config configs/dinov2/cifar-fs/test.yaml

####### Omniglot
python finetune.py --config=configs/dinov2/omniglot/finetune.yaml
python test.py --config configs/dinov2/omniglot/test.yaml
####################################################################################################################################################################################
####################################################################################################################################################################################
############################################# torchvision
####### mini-imagenet
# python finetune.py --config=configs/dinov2/mini-imagenet/finetune.yaml
# python test.py --config=configs/torchvision/mini-imagenet/test.yaml 


####### tiered-imagenet
# python finetune.py --config=configs/dinov2/tiered-imagenet/finetune.yaml
# python test.py --config configs/torchvision/tiered-imagenet/test.yaml 

####### domain-net
# python test.py --config=configs/torchvision/domain-net/test.yaml
# python finetune.py --config=configs/dinov2/domain-net/finetune.yaml

####### CIFAR-FS
python test.py --config=configs/torchvision/cifar-fs/test.yaml 
python finetune.py --config=configs/torchvision/cifar-fs/finetune_ViT.yaml
############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################################################################################################################################################
########### stdFT

# python lp_finetune.py --config=configs/clip/mini-imagenet/finetune_ViT_lp.yaml

# python test.py --config=configs/mocov3/mini-imagenet/test.yaml 
# python test.py --config=configs/mocov3/tiered-imagenet/test.yaml 
# python test.py --config=configs/mocov3/domain-net/test.yaml 
# python test.py --config=configs/dinov2/domain-net/test.yaml 
# python test.py --config=configs/torchvision/domain-net/test.yaml 

# python lp_finetune.py --config=configs/clip/tiered-imagenet/finetune_ViT_lp.yaml


# python lp_finetune.py --config=configs/clip/tiered-imagenet/finetune_RN50_lp.yaml
# python test.py --config=configs/clip/tiered-imagenet/test.yaml


# python lp_finetune.py --config=configs/clip/domain-net/finetune_ViT_lp.yaml
# python test.py --config=configs/clip/domain-net/test.yaml 



### in sinter
# python finetune.py \
#     --config=configs/dinov2/tiered-imagenet/finetune.yaml \
#     --n_shot 2 \
#     --sample_per_task 300 \
#     --n_batch_train 200

# python finetune.py \
#     --config=configs/clip/tiered-imagenet/finetune_ViT.yaml \
#     --n_shot 4 \
#     --sample_per_task 600 \
#     --n_batch_train 200


# python finetune.py \
#     --config=configs/mocov3/tiered-imagenet/finetune.yaml \
#     --n_shot 3 \
#     --sample_per_task 450 \
#     --n_batch_train 200

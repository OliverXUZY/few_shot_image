#!/usr/bin/env bash
#
#SBATCH --output=./log/VL/l_device_%j.out
#SBATCH --error=./log/VL/e_device_%j.err
#SBATCH -J train_val  # give the job a name   
#***SBATCH --partition=batch_default ***
# 
# 1 node, 1 CPU per node (total 1 CPU), wall clock time of hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=1   ## Processors per node
#SBATCH --ntasks=1            ## Tasks
#SBATCH --gres=gpu:1          ## GPUs
#SBATCH --cpus-per-task=16     ## CPUs per task; number of threads of each task
#SBATCH -t 256:00:00          ## Walltime
#SBATCH --mem=80GB
#SBATCH -p lianglab,research
#SBATCH --exclude=euler[01-16],euler[20-28]
source ~/.bashrc


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



####====================================================
## Vision language model
# python runVisionLM.py --config=configs/VL/tiered-imagenet.yaml --do_test
# python runVisionLM.py --config=configs/VL/tiered-imagenet.yaml --do_train --do_val

# python runVisionLM.py --config=configs/VL/mini-imagenet.yaml --do_train --do_val
# python runVisionLM.py --config=configs/VL/mini-imagenet.yaml --do_test


# python visionLM.py --config=configs/VL/mini-imagenet.yaml --do_train --do_val

python visionLM.py --config=configs/VL/tiered-imagenet.yaml --do_train --do_val
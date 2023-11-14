#!/usr/bin/env bash
#
#SBATCH -J vary_num  # give the job a name   
#***SBATCH --partition=batch_default ***
# 
# 1 node, 1 CPU per node (total 1 CPU), wall clock time of hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=1   ## Processors per node
#SBATCH --ntasks=1            ## Tasks
#SBATCH --gres=gpu:4          ## GPUs
#SBATCH --cpus-per-task=8     ## CPUs per task; number of threads of each task
#SBATCH -t 56:00:00          ## Walltime
#SBATCH --mem=40GB
#SBATCH -p lianglab,research
#SBATCH --exclude=euler[01-16],euler[20-28]
#SBATCH --error=./log/vision/ft_array_job_slurm_%A_%a.err
#SBATCH --output=./log/vision/ft_array_job_slurm_%A_%a.out
source ~/.bashrc

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

#*** for testing CUDA, run python code below
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
# for take different input from different lines of input_file_list.txt
# echo $( awk "NR==$SLURM_ARRAY_TASK_ID" input_path_list.txt )

# python finetune.py \
#     --config=configs/clip/mini-imagenet/finetune_RN50.yaml \
#     --n_shot $( awk "NR==$SLURM_ARRAY_TASK_ID" input_file_list.txt ) \
#     --sample_per_task $( awk "NR==$SLURM_ARRAY_TASK_ID" input_file_list2.txt ) \
#     --n_batch_train $1

# python finetune.py \
#     --config=configs/dinov2/tiered-imagenet/finetune.yaml \
#     --n_shot $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_shot.txt ) \
#     --sample_per_task $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_m.txt ) \
#     --n_batch_train $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_M.txt ) \
#     --n_episode 

python finetune.py \
    --config=configs/mocov3/tiered-imagenet/finetune.yaml \
    --n_shot 1 \
    --sample_per_task 150 \
    --n_batch_train $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_M.txt ) \
    --n_episode  $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_episode.txt )


# sbatch --array=5-16 jobArrayScript.sh

# --dependency=afterany:341497

# continue to tune
# python finetune.py \
#     --config=configs/clip/mini-imagenet/finetune_RN50.yaml \
#     --path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_path_list.txt) \
#     --n_shot 4 \
#     --n_batch_train $( awk "NR==$SLURM_ARRAY_TASK_ID" input_file_list3.txt ) \


# python finetune.py \
#     --config=configs/clip/tiered-imagenet/finetune_RN50.yaml \
#     --path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_path_list.txt)

######## test
# python test.py \
#     --config=configs/clip/mini-imagenet/test.yaml \
#     --path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_path_list.txt ) \


# python test.py \
#     --config=configs/clip/mini-imagenet/test.yaml \
#     --path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_path_list.txt ) \
#     --exp vary_num_shot


###########  torchvision
# python finetune.py --config $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_path_list.txt )

########### LP
# python lp_finetune.py --config $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_path_list.txt )
# sbatch --array=29 jobArrayScript.sh


# python finetune.py --config $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_path_list.txt )
# sbatch --array=26-28 jobArrayScript.sh 
### final running:



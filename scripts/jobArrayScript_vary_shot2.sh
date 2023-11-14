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
#SBATCH --gres=gpu:1          ## GPUs
#SBATCH --cpus-per-task=4     ## CPUs per task; number of threads of each task
#SBATCH -t 56:00:00          ## Walltime
#SBATCH --mem=40GB
#SBATCH -p lianglab,research
#SBATCH --exclude=euler[01-16],euler[20-28]
#SBATCH --error=./log/array/tiered_ViT_array_job_slurm_%A_%a.err
#SBATCH --output=./log/array/tiered_ViT_array_job_slurm_%A_%a.out
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
#     --config configs/mocov3/tiered-imagenet/finetune.yaml \
#     --n_shot $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_shot.txt ) \
#     --sample_per_task $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_m.txt ) \
#     --n_batch_train 400 \
#     --n_way 5 \
#     --output_path ./save/mocov3/tiered-imagenet/vary_num_shot \
#     --lr 2.e-5
   
### final running:
# sbatch --array=5-8  jobArrayScript_vary_shot2.sh
# --dependency=afterany:341519


########################################################################
######################## second setting 1 10 20
# python finetune.py \
#     --config configs/mocov3/tiered-imagenet/finetune.yaml \
#     --n_shot $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_shot.txt ) \
#     --sample_per_task $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_m.txt ) \
#     --n_batch_train $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_M_num_shot.txt) \
#     --n_way 5 \
#     --output_path ./save/mocov3/tiered-imagenet/vary_num_shot_setting2

# sbatch --array=11-19  jobArrayScript_vary_shot2.sh

########################################################################
######################## third setting 1 10 20
python finetune.py \
    --config configs/mocov3/tiered-imagenet/finetune.yaml \
    --n_shot $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_shot.txt ) \
    --n_query 20 \
    --n_batch_train 400 \
    --n_way 5 \
    --output_path ./save/mocov3/tiered-imagenet/vary_num_shot_setting3 \
    --lr 2.e-5

# sbatch --array=1-8 jobArrayScript_vary_shot2.sh
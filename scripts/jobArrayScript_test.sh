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
#SBATCH --cpus-per-task=8     ## CPUs per task; number of threads of each task
#SBATCH -t 56:00:00          ## Walltime
#SBATCH --mem=40GB
#SBATCH -p lianglab,research
#SBATCH --exclude=euler[24-27]
#SBATCH --error=./log/array/test_array_job_slurm_%A_%a.err
#SBATCH --output=./log/array/test_array_job_slurm_%A_%a.out
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

# python test.py \
#     --config=configs/clip/tiered-imagenet/test.yaml \
#     --path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_path_list2.txt ) \
#     --exp vary_num_shot

# python test.py --config configs/mocov3/tiered-imagenet/test.yaml \
#     --path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/test_input_path_list.txt ) \
#     --exp vary_num_shot \
#     --n_shot 1 \
#     --n_way 5

# python test.py --config configs/mocov3/tiered-imagenet/test.yaml \
#     --path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/test_input_path_list.txt ) \
#     --exp sample_complex

# python test.py --config configs/mocov3/tiered-imagenet/test.yaml \
#     --n_shot  $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_shot.txt )

### final running:
# sbatch --array=18-29 jobArrayScript_test.sh


# python test.py --config configs/mocov3/tiered-imagenet/test.yaml \
#     --path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/test_input_path_list.txt ) \
#     --exp vary_num_shot \
#     --n_shot 1 \
#     --n_way 5


# sbatch --array=31-34  jobArrayScript_test.sh
# --dependency=afterany:341631

python test.py --config configs/mocov3/tiered-imagenet/test.yaml \
    --path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/test_input_path_list.txt ) \
    --exp vary_num_shot_setting4 \
    --n_shot 1 \
    --n_way 5


# sbatch --array=2 jobArrayScript_test.sh
# sbatch --array=4-8 --dependency=afterany:342180 jobArrayScript_test.sh

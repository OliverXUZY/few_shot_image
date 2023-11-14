#!/usr/bin/env bash
#
#SBATCH -J vary_num  # give the job a name   
#***SBATCH --partition=batch_default ***
# 
# 1 node, 1 CPU per node (total 1 CPU), wall clock time of hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=4   ## Processors per node
#SBATCH --ntasks=1            ## Tasks
#SBATCH --gres=gpu:1          ## GPUs
#SBATCH --cpus-per-task=4     ## CPUs per task; number of threads of each task
#SBATCH -t 56:00:00          ## Walltime
#SBATCH --mem=40GB
#SBATCH -p lianglab
#SBATCH --exclude=euler[01-16],euler[20-28]
#SBATCH --error=./log/VL/tiered_RN_array_job_slurm_%A_%a.err
#SBATCH --output=./log/VL/tiered_RN_array_job_slurm_%A_%a.out
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
echo $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/test_input_yaml_list.txt )

python test.py --config $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/test_input_yaml_list.txt )

### final running:
# sbatch --array=10-12 jobArrayScript3.sh
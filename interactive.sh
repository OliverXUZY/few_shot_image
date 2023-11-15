# Start GPU monitoring in the background
(
    while true; do
        nvidia-smi | tee -a ./log/VL/gpu_usage_${SLURM_JOB_ID}.log
        sleep 30  # Log every 60 seconds
    done
) &
monitor_pid=$!

python finetune_maml.py \
    --config configs/dinov2/mini-imagenet/finetune_maml.yaml  \
    --output_path ./save/dinov2/mini-imagenet/

# Kill the GPU monitoring process
kill $monitor_pid

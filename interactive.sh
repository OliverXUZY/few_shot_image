python finetune_maml.py \
    --config configs/clip/tiered-imagenet/finetune_ViT_maml.yaml  \
    --output_path ./save/clip/tiered-imagenet/

python test.py --config configs/clip/tiered-imagenet/test.yaml

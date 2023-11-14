
# for cls in 175 43 10 
# do
#     python finetune.py --config configs/clip/tiered-imagenet/finetune_ViT.yaml --limited_class $cls
# done

# python finetune.py --config configs/clip/tiered-imagenet/finetune_ViT.yaml --limited_class 10

# python finetune.py --config configs/torchvision/mini-imagenet/finetune_RN18.yaml

# python test.py --config=configs/torchvision/mini-imagenet/test.yaml 


# #### clip
# python finetune.py --config=configs/clip/mini-imagenet/finetune_ViT.yaml
# python test.py --config configs/clip/mini-imagenet/test.yaml

python test.py --config configs/clip/domain-net/test.yaml

python lp_finetune.py --config configs/clip/mini-imagenet/finetune_ViT_lp.yaml

# python finetune.py --config=configs/clip/domain-net/finetune_ViT.yaml
# python test.py --config configs/clip/mixed-set/test.yaml

# python finetune.py --config configs/clip/mixed-set/finetune.yaml

# #### dinov2
# python finetune.py --config configs/dinov2/mixed-set/finetune.yaml

# python test.py --config configs/dinov2/mixed-set/test.yaml

python test.py --config configs/dinov2/mini-imagenet/test.yaml


# ## sup
# python finetune.py --config configs/torchvision/mixed-set/finetune.yaml

# python test.py --config configs/torchvision/mixed-set/test.yaml

python test.py --config configs/torchvision/tiered-imagenet/test.yaml
for cls in 175 43 10 
do
    python finetune.py --config configs/torchvision/tiered-imagenet/finetune_ViT.yaml --limited_class $cls
done
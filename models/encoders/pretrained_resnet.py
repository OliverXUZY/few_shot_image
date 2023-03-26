
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import clip

from .encoders import register
from ..modules import *

import torchvision

__all__ = ['RN50', 'ResNet50_mocov2']

@register('RN50')
class RN50(Module):
    '''
    ResNet50 encoder pre-trained by CLIP
    '''
    def __init__(self):
        super(RN50, self).__init__()

        self.out_dim = 1024
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("RN50", device=device)
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, 1024]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model.encode_image(x)

@register('ResNet50_mocov2') # input image size for moco is still [3,224,224]: https://github.com/facebookresearch/moco/blob/5a429c00bb6d4efdf511bf31b6f01e064bf929ab/main_lincls.py#L372
class ResNet50_mocov2(Module):
    '''
    ResNet50 encoder pre-trained by moco v2
    '''
    def __init__(self, ckpt_path):
        super(ResNet50_mocov2, self).__init__()
        self.model = torchvision.models.__dict__['resnet50']()
        self.model.fc = nn.Identity() # replace last nn.Linear(2048, 1000) to Identity()

        self.out_dim = 2048

        ## load state_dict of moco v2
        print("=> loading checkpoint '{}'".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # rename moco pre-trained keys
        state_dict = ckpt["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        self.model.load_state_dict(state_dict, strict=False)
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, 2048]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model(x)








import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import clip

from .encoders import register
from ..modules import *

__all__ = ['ViT-B32']

@register('ViT-B32')
class ViTB32(Module):
    '''
    ViT encoder pre-trained by CLIP
    '''
    def __init__(self):
        super(ViTB32, self).__init__()

        self.out_dim = 512
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
    
    def get_out_dim(self):
        return self.out_dim
    
    def forward(self, x):
        '''
        input size: [Batch_size, 3, 224, 224]
        output size: [Batch_size, 512]
        '''
        assert x.dim() == 4        # [Batch_size, 3, 224, 224]

        return self.model.encode_image(x)









import argparse
import os
import random
import sys
sys.path.append("/srv/home/zxu444/few_shot_image")
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
from models import encoders, classifiers
import utils
import utils.optimizers as optimizers

import clip

templates = ['a photo of a {}',
             'itap of a {}.',
            'a bad photo of the {}.',
            'a origami {}.',
            'a photo of the large {}.',
            'a {} in a video game.',
            'art of the {}.',
            'a photo of the small {}.']

def encode_name_to_feature(name, clip_model):
    '''
    name: str: 'electric guitar'

    clip_model: model.encode_text in CLIP model (enc.model)
    '''
    text_embedding_templates = list(map(
        lambda template: clip_model.encode_text(clip.tokenize(
        template.format(name)                  # [1,embdeeing_size] [1,77]
        ).cuda(non_blocking=True)),            # [1,D]   [1,512]
        templates
    ))
    text_embedding_templates = torch.concat(text_embedding_templates)
    print(text_embedding_templates.shape)
    pass



def text_encoder(shot_names, clip_model):
    '''
    shot_names: list with length Y, each element of list is a tuple with E names
    # [('vase', 'Alaskan Malamute', 'electric guitar', 'mixing bowl'),
    # ('red king crab', 'scoreboard', 'Dalmatian', 'Golden Retriever'),
    # ('trifle', 'lion', 'vase', 'red king crab'),
    # ('black-footed ferret', 'crate', 'nematode', 'front curtain'),
    # ('crate', 'Golden Retriever', 'bookstore', 'Dalmatian')]

    clip_model: model.encode_text in CLIP model (enc.model)
    '''
    s = []
    for template in templates:
        token_ep = list(map(
            lambda x: clip_model.encode_text(     # x = ('vase', 'Alaskan Malamute', 'electric guitar', 'mixing bowl')
                torch.concat(
                    [clip.tokenize(template.format(name)) for name in x]   # each element is [1,77] tokens for one sentence
                    ).cuda(non_blocking=True)                               # 4 eps ('vase', 'Alaskan Malamute', 'electric guitar', 'mixing bowl') for tokens for one of Y class [E,77] [4, 77]
                ),        # 4 eps for text features for one of Y class  [E,D] [4, 512]
            shot_names
            ))            # list with length Y, each element is above [E,D]

        textfea_ep = torch.stack(token_ep)      # [Y,E,D] [5, 4, 512]
        s.append(textfea_ep)
    s = torch.stack(s)                      # [T=8,Y,E,D] 8 templates        [8, 5, 4, 512]
    s = s.transpose(0,2)                    # [E,Y,T,D]      [4, 5, 8, 512]
    s /= s.norm(dim = -1, keepdim=True)     # normalize

    return s

def main():
    enc = encoders.make("ViT-B32")
    encode_name_to_feature('vase', enc.model)
    


if __name__ == "__main__":
    main()
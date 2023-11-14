import os
import pickle
import time
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import *
from torchvision.datasets import ImageFolder

@register('meta-domain-net') 
class MetaDomainNet(Dataset):
  def __init__(self, root, split='train', size=224, transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, deterministic = False):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      transform (str): data augmentation. Default: None
    """
    super(MetaDomainNet, self).__init__()

    split_dict = {'train': 'train',        # standard train
                  'val': 'val',            # standard val
                  'test': 'test',          # standard test
                  'meta-train': 'train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
    split_tag = split_dict.get(split) or split
    split_dir = '{}/{}'.format(root, split_tag)
    print(split_dir)
    assert os.path.isdir(split_dir)
    
    # self.statistics = {'mean': [0.471, 0.450, 0.403],
    #                    'std':  [0.278, 0.268, 0.284]}
    self.statistics = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    
    self.transform = get_transform(transform, size, self.statistics)

    self.dataset = ImageFolder(root = split_dir)

    if split_tag == "train":
      n_class = 180
    elif split_tag == "val":
      n_class = 65
    else:
      n_class = 100
    self.n_class = n_class

    ### sampling part
    print("start sampling part dataset")
    ##### cache label file since it's time consuming
    cache_label_file = os.path.join(root,"cached_{}_labels_domainNet.npy".format(split_tag))
    if os.path.exists(cache_label_file):
      start = time.time()
      self.label = np.load(cache_label_file)
      print(
            "Loading labels from cached file {} [took {:.3f} s]".format(cache_label_file, time.time() - start)
      )
    else:
      print(f"Creating labels from dataset file at {root}")
      start = time.time()
      self.label = np.array([target for _, target in self.dataset])
      np.save(cache_label_file, self.label)
      # ^ This seems to take forever (but 5 mins at my laptop) so I want to investigate why and how we can improve.
      print(
            "Saving labels into cached file {} [took {:.3f} m]".format(cache_label_file, (time.time() - start)/60)
      )

    self.catlocs = tuple()
    for cat in range(self.n_class):
      self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)

    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query
    self.deterministic = deterministic
  
  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    if self.deterministic:
      np.random.seed(index)  ## add for control # of tasks and # of images
    s, q = self.n_shot, self.n_query
    sv, qv = 1, 1
    shot, query = tuple(), tuple()
    
    cats = np.random.choice(self.n_class, self.n_way, replace=False)
    for c in cats:
      idx = np.random.choice(self.catlocs[c], sv * s + qv * q, replace=False)      # random choose n_shot*shot_view + n_query*query_view (1*1+15*1) images in each classes
      s_idx, q_idx = idx[:sv * s], idx[-qv * q:]
      # print([self.transform(self.dataset[i][0]).shape for i in s_idx])
      # print([self.transform(self.dataset[i][0]).shape for i in q_idx])
      c_shot = torch.stack([self.transform(self.dataset[i][0]) for i in s_idx])          # [5(1),3,84,84] [S*SV, C, H ,W]
      c_query = torch.stack([self.transform(self.dataset[i][0]) for i in q_idx])         # [15,3,84,84] [Q*QV, C, H ,W]
      c_shot = c_shot.view(sv, s, 1, *c_shot.shape[-3:])   # hard code V = 1             # [1,5(1),1,3,84,84] [SV, S, V, C, H ,W]
      c_query = c_query.view(qv, q, *c_query.shape[-3:])                           # [1,15,3,84,84] [QV, Q, C, H ,W]
      shot += (c_shot,)
      query += (c_query,)
    
    shot = torch.cat(shot, dim=1)      # [SV, Y * S, V, C, H, W]                   # [1, 5, 1, 3, 84, 84] 
    query = torch.cat(query, dim=1)    # [QV, Y * Q, C, H, W]                      # [1, 50, 3, 84, 84]
    cats = torch.from_numpy(cats)
    return shot, query, cats
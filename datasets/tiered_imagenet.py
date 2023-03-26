import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import *

import time
import logging
logger = logging.getLogger(__name__)

@register('tiered-imagenet')
class TieredImageNet(Dataset):
  def __init__(self, root, split='train', size=84, n_view=1, transform=None):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      n_view (int): number of augmented views. Default: 1
      transform (str): data augmentation. Default: None
    """
    super(TieredImageNet, self).__init__()
    
    split_dict = {'train': 'train',         # standard train
                  'val': 'train_phase_val', # standard val
                  'meta-train': 'train',    # meta-train
                  'meta-val': 'val',        # meta-val
                  'meta-test': 'test',      # meta-test
                 }
    split_tag = split_dict[split]

    split_file = os.path.join(root, split_tag + '_images.npz')
    label_file = os.path.join(root, split_tag + '_labels.pkl')
    assert os.path.isfile(split_file)
    assert os.path.isfile(label_file)
    data = np.load(split_file, allow_pickle=True)['images']
    data = data[:, :, :, ::-1]
    with open(label_file, 'rb') as f:
      label = pickle.load(f)['labels']

    data = [Image.fromarray(x) for x in data]
    label = np.array(label)
    label_key = sorted(np.unique(label))
    label_map = dict(zip(label_key, range(len(label_key))))
    new_label = np.array([label_map[x] for x in label])

    self.root = root
    self.split_tag = split_tag
    self.size = size
    
    self.data = data
    self.label = new_label
    self.n_class = len(label_key)

    self.statistics = {'mean': [0.478, 0.456, 0.410],
                       'std':  [0.279, 0.274, 0.286]}
    transform = get_transform(transform, size, self.statistics)
    self.transform = MultiViewTransform(transform, n_view)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image = self.transform(self.data[index])              # [V, C, H, W]
    label = self.label[index]
    return image, label


@register('meta-tiered-imagenet')
class MetaTieredImageNet(TieredImageNet):
  def __init__(self, root, split='meta-train', size=84, 
               n_view=1, n_meta_view=1, share_query=False,
               transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, deterministic = False):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      n_view (int): number of augmented views of image. Default: 1
      n_meta_view (int): number of augmented views of task. Default: 1
      share_query (bool): True if use distinct query set for each meta-view. 
        Default: False
      transform (str): training data augmentation. Default: None
      val_transform (str): validation data augmentation. Default: None
      n_batch (int): number of mini-batches per epoch. Default: 200
      n_episode (int): number of episodes (tasks) per mini-batch. Default: 4
      n_way (int): number of categories per episode. Default: 5
      n_shot (int): number of training (support) samples per category. 
        Default: 1
      n_query (int): number of validation (query) samples per category. 
        Default: 15
      deterministic: whether set images in dataset to be deterministic in each epoch
    """
    super(MetaTieredImageNet, self).__init__(root, split, size, n_view, transform)
    
    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query
    self.n_shot_view = self.n_meta_view = n_meta_view
    if share_query:
      self.n_query_view = 1
    else:
      self.n_query_view = n_meta_view

    self.catlocs = tuple()
    for cat in range(self.n_class):
      self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)

    self.val_transform = get_transform(val_transform, size, self.statistics)

    self.deterministic = deterministic

  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    if self.deterministic:
      np.random.seed(index)  ## add for control # of tasks and # of images
    s, q = self.n_shot, self.n_query
    sv, qv = self.n_shot_view, self.n_query_view
    shot, query = tuple(), tuple()
    
    cats = np.random.choice(self.n_class, self.n_way, replace=False)
    for c in cats:
      idx = np.random.choice(self.catlocs[c], sv * s + qv * q, replace=False)
      s_idx, q_idx = idx[:sv * s], idx[-qv * q:]
      c_shot = torch.stack([self.transform(self.data[i]) for i in s_idx])
      c_query = torch.stack([self.val_transform(self.data[i]) for i in q_idx])
      c_shot = c_shot.view(sv, s, *c_shot.shape[-4:])
      c_query = c_query.view(qv, q, *c_query.shape[-3:])
      shot += (c_shot,)
      query += (c_query,)
    
    shot = torch.cat(shot, dim=1)      # [SV, Y * S, V, C, H, W]
    query = torch.cat(query, dim=1)    # [QV, Y * Q, C, H, W]
    cats = torch.from_numpy(cats)
    return shot, query, cats

######################################################################################################################
########################################## vision language dataset
from torchvision.datasets import ImageFolder

data_root = "/srv/home/zxu444/datasets"
with open(os.path.join(data_root,'classnames.txt')) as f:
    lines = [line.rstrip() for line in f]

class_to_name = {}
for line in lines:
    s_id = line.find(' ')
    class_to_name[line[:s_id]] = line[s_id+1:]

@register('vl-meta-tiered-imagenet') 
class VLMetaTieredImageNet(Dataset):
  def __init__(self, root, split='train', size=224, transform=None,
               n_batch=200, n_episode=4, n_way=15, n_query=15, deterministic = False):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      transform (str): data augmentation. Default: None
    """
    super(VLMetaTieredImageNet, self).__init__()

    split_dict = {'train': 'train',        # standard train
                  'val': 'val',            # standard val
                  'test': 'test',          # standard test
                  'meta-train': 'train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
    split_tag = split_dict.get(split) or split
    split_dir = '{}/{}'.format(root, split_tag)
    assert os.path.isdir(split_dir)
    
    self.statistics = {'mean': [0.471, 0.450, 0.403],
                       'std':  [0.278, 0.268, 0.284]}
    self.transform = get_transform(transform, size, self.statistics)

    self.dataset = ImageFolder(root = split_dir, transform = self.transform)

    idx_to_name = {}
    for c in self.dataset.class_to_idx:
        idx_to_name[self.dataset.class_to_idx[c]] = class_to_name[c]
    self.n_class = len(idx_to_name)
    self.label_idx_to_name = idx_to_name

    
    ### sampling part
    print("start sampling part dataset")
    ##### cache label file since it's time consuming
    cache_label_file = os.path.join(root,"cached_{}_labels_vl-tiered-imagenet.npy".format(split_tag))
    if os.path.exists(cache_label_file):
      start = time.time()
      self.label = np.load(cache_label_file)
      print(
          f"Loading labels from cached file {cache_label_file} [took %.3f s]", time.time() - start
      )
    else:
      print(f"Creating labels from dataset file at {root}")
      start = time.time()
      self.label = np.array([target for _, target in self.dataset])
      np.save(cache_label_file, self.label)
      # ^ This seems to take a lot of time(even longer than my laptop) so I want to investigate why and how we can improve.
      print(
          "Saving labels into cached file %s [took %.3f s]", cache_label_file, time.time() - start
      )

    self.catlocs = tuple()
    for cat in range(self.n_class):
      self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)

    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_query = n_query
    self.deterministic = deterministic
  
  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    if self.deterministic:
      np.random.seed(index) 
    q = self.n_query
    query = tuple()

    cats = np.random.choice(self.n_class, self.n_way, replace=False)
    label_names = []
    for c in cats:
      label_names.append(self.label_idx_to_name[c])
      idx = np.random.choice(self.catlocs[c], q, replace=False) 
      c_query = torch.stack([self.dataset[i][0] for i in idx])  # [q, C, H ,W] [3, 3, 224, 224]
      query += (c_query,)
    query = torch.cat(query)    # [QV, Y * Q, C, H, W] 
    cats = torch.from_numpy(cats)
    
    return query, cats, label_names
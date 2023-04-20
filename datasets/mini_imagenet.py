import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import *


@register('mini-imagenet')
class MiniImageNet(Dataset):
  def __init__(self, root, split='train', size=84, n_view=1, transform=None):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      n_view (int): number of augmented views. Default: 1
      transform (str): data augmentation. Default: None
    """
    super(MiniImageNet, self).__init__()
    
    split_dict = {'train': 'train_phase_train',        # standard train
                  'val': 'train_phase_val',            # standard val
                  'trainval': 'train_phase_trainval',  # standard train and val
                  'test': 'train_phase_test',          # standard test
                  'meta-train': 'train_phase_train',   # meta-train
                  'meta-val': 'val',                   # meta-val
                  'meta-test': 'test',                 # meta-test
                 }
    split_tag = split_dict.get(split) or split
    # split_tag = split_dict[split]
    split_file = '{}/miniImageNet_category_split_{}.pickle'.format(root, split_tag)
    print(split_file)

    # split_file = os.path.join(root, split_tag + '.pickle')
    assert os.path.isfile(split_file)
    with open(split_file, 'rb') as f:
      pack = pickle.load(f, encoding='latin1')
    data, label = pack['data'], pack['labels']

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
    
    self.statistics = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    transform = get_transform(transform, size, self.statistics)
    self.transform = MultiViewTransform(transform, n_view)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image = self.transform(self.data[index])            # [V, C, H, W]
    label = self.label[index]
    return image, label


@register('meta-mini-imagenet')
class MetaMiniImageNet(MiniImageNet):
  def __init__(self, root, split='meta-train', size=84, 
               n_view=1, n_meta_view=1, share_query=False,
               transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15, deterministic = False):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      n_view (int): number of augmented views of image. Default: 1. Only for shot, used in MultiViewTransform.
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
    super(MetaMiniImageNet, self).__init__(root, split, size, n_view, transform)
    
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
      idx = np.random.choice(self.catlocs[c], sv * s + qv * q, replace=False)      # random choose n_shot*shot_view + n_query*query_view (1*1+15*1) images in each classes
      s_idx, q_idx = idx[:sv * s], idx[-qv * q:]
      c_shot = torch.stack([self.transform(self.data[i]) for i in s_idx])          # [1,1,3,84,84] [S*SV, V, C, H ,W]
      c_query = torch.stack([self.val_transform(self.data[i]) for i in q_idx])     # [15,3,84,84] [Q*QV, C, H ,W]
      c_shot = c_shot.view(sv, s, *c_shot.shape[-4:])                              # [1,1,1,3,84,84] [SV, S, V, C, H ,W]
      c_query = c_query.view(qv, q, *c_query.shape[-3:])                           # [1,10,3,84,84] [QV, Q, C, H ,W]
      shot += (c_shot,)
      query += (c_query,)
    
    shot = torch.cat(shot, dim=1)      # [SV, Y * S, V, C, H, W]                   # [1, 5, 1, 3, 84, 84] 
    query = torch.cat(query, dim=1)    # [QV, Y * Q, C, H, W]                      # [1, 50, 3, 84, 84]
    cats = torch.from_numpy(cats)
    return shot, query, cats


@register('unsup-meta-mini-imagenet')
class UnsupMetaMiniImageNet(MiniImageNet):
  '''
  This class is for unsup finetuneing
  '''
  def __init__(self, root, split='meta-train', size=84, 
               n_view=1, transform=None, 
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      n_view (int): number of augmented views of image. Default: 1
      transform (str): training data augmentation. Default: None
      n_batch (int): number of mini-batches per epoch. Default: 200
      n_episode (int): number of episodes (tasks) per mini-batch. Default: 4
      n_way (int): number of categories per episode. Default: 5
      n_shot (int): number of training (support) samples per category. 
        Default: 1
      n_query (int): number of validation (query) samples per category. 
        Default: 15
    """
    super(UnsupMetaMiniImageNet, self).__init__(root, split, size, n_view, transform)
    
    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query

    self.catlocs = tuple()
    for cat in range(self.n_class):
      self.catlocs += (np.argwhere(self.label == cat).reshape(-1),)

  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    s, q = self.n_shot, self.n_query
    images = tuple()
    
    cats = np.random.choice(self.n_class, self.n_way, replace=False)
    for c in cats:
      idx = np.random.choice(self.catlocs[c], s + q, replace=False)

      image = torch.stack([self.transform(self.data[i]) for i in idx])  # [11, 2, 3, 84, 84] [S+Q, n_view = V, C,H,W]
      images += (image, )

      
    images = torch.cat(images, dim=0)      # [(S+Q)*Y, n_view = V, C,H,W] [55, 2, 3, 84, 84]
    return images
  

########################################## vision language dataset
from torchvision.datasets import ImageFolder
idx_to_name ={ 
"test": {0: 'nematode',
 1: 'red king crab',
 2: 'Golden Retriever',
 3: 'Alaskan Malamute',
 4: 'Dalmatian',
 5: 'African wild dog',
 6: 'lion',
 7: 'ant',
 8: 'black-footed ferret',
 9: 'bookstore',
 10: 'crate',
 11: 'cuirass',
 12: 'electric guitar',
 13: 'hourglass',
 14: 'mixing bowl',
 15: 'school bus',
 16: 'scoreboard',
 17: 'front curtain',
 18: 'vase',
 19: 'trifle'},

 "val":{0: 'goose',
 1: 'Ibizan Hound',
 2: 'Alaskan tundra wolf',
 3: 'meerkat',
 4: 'rhinoceros beetle',
 5: 'cannon',
 6: 'cardboard box / carton',
 7: 'catamaran',
 8: 'combination lock',
 9: 'garbage truck',
 10: 'gymnastic horizontal bar',
 11: 'iPod',
 12: 'miniskirt',
 13: 'missile',
 14: 'poncho',
 15: 'coral reef'},

 "train": {0: 'house finch',
 1: 'American robin',
 2: 'triceratops',
 3: 'green mamba',
 4: 'harvestman',
 5: 'toucan',
 6: 'jellyfish',
 7: 'dugong',
 8: 'Treeing Walker Coonhound',
 9: 'Saluki',
 10: 'Gordon Setter',
 11: 'Komondor',
 12: 'Boxer',
 13: 'Tibetan Mastiff',
 14: 'French Bulldog',
 15: 'Newfoundland dog',
 16: 'Miniature Poodle',
 17: 'Arctic fox',
 18: 'ladybug',
 19: 'three-toed sloth',
 20: 'rock beauty fish',
 21: 'aircraft carrier',
 22: 'trash can',
 23: 'barrel',
 24: 'beer bottle',
 25: 'carousel',
 26: 'bell or wind chime',
 27: 'clogs',
 28: 'cocktail shaker',
 29: 'dishcloth',
 30: 'dome',
 31: 'filing cabinet',
 32: 'fire screen',
 33: 'frying pan',
 34: 'hair clip',
 35: 'holster',
 36: 'lipstick',
 37: 'oboe',
 38: 'pipe organ',
 39: 'parallel bars',
 40: 'pencil case',
 41: 'photocopier',
 42: 'prayer rug',
 43: 'fishing casting reel',
 44: 'slot machine',
 45: 'snorkel',
 46: 'solar thermal collector',
 47: 'spider web',
 48: 'stage',
 49: 'tank',
 50: 'tile roof',
 51: 'tobacco shop',
 52: 'unicycle',
 53: 'upright piano',
 54: 'wok',
 55: 'split-rail fence',
 56: 'sailboat',
 57: 'traffic or street sign',
 58: 'consomme',
 59: 'hot dog',
 60: 'orange',
 61: 'cliff',
 62: 'bolete',
 63: 'corn cob'}
}
@register('vl-meta-mini-imagenet') 
class VLMetaMiniImageNet(Dataset):
  def __init__(self, root, split='train', size=224, transform=None,
               n_batch=200, n_episode=4, n_way=5, n_query=15, deterministic = False):
    """
    Args:
      root (str): root path of dataset.
      split (str): dataset split. Default: 'train'
      size (int): image resolution. Default: 84
      transform (str): data augmentation. Default: None
    """
    super(VLMetaMiniImageNet, self).__init__()

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
    
    self.statistics = {'mean': [0.471, 0.450, 0.403],
                       'std':  [0.278, 0.268, 0.284]}
    self.transform = get_transform(transform, size, self.statistics)

    self.dataset = ImageFolder(root = split_dir, transform = self.transform)
    self.label_idx_to_name = idx_to_name[split_tag]

    ### sampling part
    self.catlocs = np.arange(len(self.dataset)).reshape(-1,600)

    self.n_class = self.catlocs.shape[0]
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
    query = torch.cat(query)    # [Y * Q, C, H, W] 
    cats = torch.from_numpy(cats)
    
    return query, cats, label_names
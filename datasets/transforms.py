import random

import torch
import torchvision.transforms as transforms
from PIL import ImageFilter


class MultiViewTransform(object):
  def __init__(self, transform, n_view=2):
    self.transform = transform
    self.n_view = n_view

  def __call__(self, x):
    views = torch.stack([self.transform(x) for _ in range(self.n_view)])
    return views


class GaussianBlur(object):
  def __init__(self, sigma=(.1, 2.)):
    self.sigma = sigma

  def __call__(self, x):
    sigma = random.uniform(self.sigma[0], self.sigma[1])
    x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
    return x


def get_transform(name, size, statistics=None):
  if statistics is None:
    statistics = {'mean': [0., 0., 0.],
                  'std':  [1., 1., 1.]}
                  
  if name in ['ucb', 'ucb-fs']:
    return transforms.Compose([
      transforms.RandomResizedCrop(size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'mit':
    return transforms.Compose([
      transforms.RandomCrop(size, padding=(8 if size > 32 else 4)),
      transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'mit-fs':
    return transforms.Compose([
      transforms.RandomCrop(size, padding=(8 if size > 32 else 4)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'contrast':
    return transforms.Compose([
      transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomApply([
        transforms.ColorJitter(
          brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
      ], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      # transforms.RandomApply([GaussianBlur()], p=0.5),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'contrast-fs':
    return transforms.Compose([
      transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'flip':
    return transforms.Compose([
      transforms.Resize(size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif name == 'enlarge':
    return transforms.Compose([
      transforms.Resize(int(size * 256 / 224)),
      transforms.CenterCrop(size),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  elif  name is None:
    return transforms.Compose([
      transforms.Resize(size),
      transforms.ToTensor(),
      transforms.Normalize(**statistics),
    ])
  else:
    raise ValueError('invalid transform: {}'.format(name))
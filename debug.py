import argparse
import os
import random
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import datasets
import models
from models import encoders, classifiers
import utils
import utils.optimizers as optimizers


def main(config):
  SEED = config.get('seed') or 0
  utils.log("seed: {}".format(SEED))
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  
  # torch.backends.cudnn.enabled = True
  # torch.backends.cudnn.benchmark = True
  # torch.backends.cudnn.deterministic = True

  ##### Dataset #####
  
  # V = config['train_set_args']['n_view'] = config['V']
  # SV = config['train_set_args']['n_meta_view'] = 1
  V = SV = 1

  # meta-train
  train_set = datasets.make(config['dataset'], **config['train_set_args'])
#   utils.log('meta-train dataset: split-{} {} (x{}), {}'.format(config['train_set_args']['split'],
#     train_set[0][0].shape, len(train_set), train_set.n_class))
  
  TE = train_set.n_episode
  TY = train_set.n_way
  TS = train_set.n_shot
  TQ = train_set.n_query

  # query-set labels
  y = torch.arange(TY)[:, None]
  y = y.repeat(TE, SV, TQ).flatten()      # [TE * SV * TY * TQ]
  y = y.cuda()

  train_loader = DataLoader(train_set, TE, num_workers=8, pin_memory=True)

  for i in range(200):
    train_set[i]
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', 
                      help='configuration file')
  parser.add_argument('--tag', 
                      help='auxiliary information', 
                      type=str, default=None)
  parser.add_argument('--gpu', 
                      help='gpu device number', 
                      type=str, default='0')
  parser.add_argument('--efficient', 
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  parser.add_argument('--n_batch_train',
                      help='modify batch train num batch',
                      type=int)
  parser.add_argument('--n_shot',
                      help='num shot',
                      type=int)
  parser.add_argument('--sample_per_task',
                      help='sample_per_task',
                      type=int)
  parser.add_argument('--path', 
                      help='the path to saved model', 
                      type=str)
  parser.add_argument('--stdFT', 
                      default=False,
                      help='whether we use standard finetune', 
                      action='store_true')
  
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if args.path:
    if "tiered" in config['dataset']:
      config['path'] = "./save/tiered-imagenet/Mm_trend/{}".format(args.path)
      utils.log("load model from path: {}".format(config['path']))
      config['train_set_args']['n_way'] = int(args.path[38:40])
      args.n_shot = int(args.path[41:42])
      args.sample_per_task = int(args.path[44:47])
      args.n_batch_train = int(args.path[49:52])
    elif "mini" in config['dataset']:
      config['path'] = "./save/mini-imagenet/Mm_trend/{}".format(args.path)
      utils.log("load model from path: {}".format(config['path']))

  if args.n_batch_train:
    config['train_set_args']['n_batch'] = int(args.n_batch_train)
  if args.n_shot:
    config['train_set_args']['n_shot'] = int(args.n_shot)
  if args.sample_per_task:
    config['train_set_args']['n_query'] = int(args.sample_per_task/config['train_set_args']['n_way'] - args.n_shot)
  
    
  utils.log('{}y{}s_{}m_{}M'.format(
    config['train_set_args']['n_way'], 
    config['train_set_args']['n_shot'], 
    (config['train_set_args']['n_shot'] + config['train_set_args']['n_query']) * config['train_set_args']['n_way'],
    config['train_set_args']['n_batch']
    ))
  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu
  # config['stdFT'] = args.stdFT
  if config.get('LP'):
    print("Linear probing: LP: ".format(config['LP']))
  else:
    config['LP'] = False

  # utils.set_gpu(args.gpu)
  main(config)
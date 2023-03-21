import argparse
import os
import random

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

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
  
  V = config['train_set_args']['n_view'] = config['V']
  SV = config['train_set_args']['n_meta_view'] = 1

  # meta-train
  train_set = datasets.make(config['dataset'], **config['train_set_args'])
  utils.log('meta-train dataset: split-{} {} (x{}), {}'.format(config['train_set_args']['split'],
    train_set[0][0].shape, len(train_set), train_set.n_class))
  
  TE = train_set.n_episode
  TY = train_set.n_way
  TS = train_set.n_shot
  TQ = train_set.n_query

  # query-set labels
  y = torch.arange(TY)[:, None]
  y = y.repeat(TE, SV, TQ).flatten()      # [TE * SV * TY * TQ]
  y = y.cuda()

  train_loader = DataLoader(train_set, TE, num_workers=1, pin_memory=True)

  # meta-val
  eval_val = False
  if config.get('val_set_args'):
    eval_val = True
    val_set = datasets.make(config['dataset'], **config['val_set_args'])
    utils.log('meta-val dataset: {} (x{}), {}'.format(
      val_set[0][0].shape, len(val_set), val_set.n_class))

    E = val_set.n_episode
    Y = val_set.n_way
    S = val_set.n_shot
    Q = val_set.n_query

    # query-set labels
    val_y = torch.arange(Y)[:, None]
    val_y = val_y.repeat(E, Q).flatten()  # [E * Y * Q]
    val_y = val_y.cuda()

    val_loader = DataLoader(val_set, E, num_workers=1, pin_memory=True)
  
  ##### Model and Optimizer #####

  if config.get('path'):
    start_epoch_from = config['start_epoch_from']
    utils.log("continue to tune {} from {}".format(config['encoder'], start_epoch_from))
    assert os.path.exists(os.path.join(config['path'], config['ckpt']))
    ckpt = torch.load(os.path.join(config['path'], config['ckpt']))
    enc = encoders.load(ckpt)
  else:
    start_epoch_from = 0
    config['encoder_args'] = config.get('encoder_args') or dict()
    enc = encoders.make(config['encoder'], **config['encoder_args'])
    ckpt = {
      'encoder': config['encoder'],
      'encoder_args':  config['encoder_args'],
    }

  config['classifier_args'] = config.get('classifier_args') or dict()
  config['classifier_args']['in_dim'] = enc.get_out_dim()
  clf = classifiers.make(config['classifier'], **config['classifier_args'])
  
  model = models.Model(enc, clf)
    

  optimizer = optimizers.make(config['optimizer'], model.parameters(), 
                              **config['optimizer_args'])

  start_epoch = 1
  max_va = 0.

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.count_params(model)))
  utils.log('M: {}, m: {}'.format(config['train_set_args']['n_batch'], 
                                  (config['train_set_args']['n_shot'] + config['train_set_args']['n_query'])*config['train_set_args']['n_way']
                                  ))
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

  ckpt_name = '{}_{}_{}_{}y{}s_{}m_{}M'.format(
    config['dataset'], ckpt['encoder'], config['classifier'],
    config['train_set_args']['n_way'], config['train_set_args']['n_shot'], 
    (config['train_set_args']['n_shot'] + config['train_set_args']['n_query']) * config['train_set_args']['n_way'],
    config['train_set_args']['n_batch']
  )
  if args.tag is not None:
    ckpt_name += '[' + args.tag + ']'

  if config.get('save_path'):
    ckpt_path = os.path.join(config['save_path'], ckpt_name)
  else:
    ckpt_path = os.path.join('./save/clip', ckpt_name)
  if not config.get('path'):
    utils.ensure_path(ckpt_path)
  utils.set_log_path(ckpt_path)
  writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
  yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

  ##### Training and evaluation #####
  
  xent_loss = nn.CrossEntropyLoss().cuda()

  aves_keys = ['tl', 'ta', 'vl', 'va']
  trlog = dict()
  for k in aves_keys:
    trlog[k] = []

  # sets warmup schedule
  if config['optimizer_args'].get('warmup'):
    try:
      warmup_epochs = config['optimizer_args']['warmup_epochs']
      warmup_from = config['optimizer_args']['warmup_from']
      warmup_to = config['optimizer_args'].get('warmup_to')
    except:
      raise ValueError('must specify `warmup_epochs` and `warmup_from`.')
    if warmup_to is None:
      warmup_to = utils.decay_lr(
        warmup_epochs, config['n_epochs'], **config['optimizer_args'])
    utils.log('warm-up learning rate for {} epochs from {} to {}'.format(
      str(warmup_epochs), warmup_from, warmup_to))
  else:
    warmup_epochs = -1
    warmup_from = warmup_to = None
  
  for epoch in range(start_epoch, config['n_epochs'] + 1):
    timer_epoch.start()
    aves = {k: utils.AverageMeter() for k in aves_keys}

    model.train() 
    ## change BatchNorm:
    if "RN" in ckpt['encoder']:
        enc.apply(utils.set_bn_eval)
    np.random.seed(epoch + SEED)

    # adjust learning rate
    lr = utils.decay_lr(epoch, config['n_epochs'], **config['optimizer_args'])
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr

    for idx, (s, q, _) in enumerate(
      tqdm(train_loader, desc='train', leave=False)):
      # warm up learning rate
      if epoch <= warmup_epochs:
        lr = utils.warmup(warmup_from, warmup_to, 
                          epoch, warmup_epochs, idx, len(train_loader))
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr

      s = s.cuda(non_blocking=True)             # [TE, SV, TY * TS, V, C, H, W]
      q = q.cuda(non_blocking=True)             # [TE, SV, TY * TQ, C, H, W]
      s = s.view(TE, SV, TY, TS, *s.shape[-4:]) # [TE, SV, TY, TS, V, C, H, W]
      
      logits, _ = model(s, q)
      logits = logits.flatten(0, -2)            # [TE * SV * TY * TQ, TY]
      loss = xent_loss(logits, y)
      acc = utils.accuracy(logits, y)
      aves['tl'].update(loss.item())
      aves['ta'].update(acc[0])

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    # meta-val
    if eval_val:
      model.eval()
      np.random.seed(SEED)
    
      with torch.no_grad():
        for (s, q, _) in tqdm(val_loader, desc='val', leave=False):
          s = s.cuda(non_blocking=True)         # [E, 1, Y * S, 1, C, H, W]
          q = q.cuda(non_blocking=True)         # [E, 1, Y * Q, C, H, W]
          s = s.view(E, 1, Y, S, *s.shape[-4:]) # [E, 1, Y, S, 1, C, H, W]
          
          logits, _ = model(s, q)               # [E, 1, Y * Q, Y]
          logits = logits.flatten(0, -2)        # [E * Y * Q, Y]
          loss = xent_loss(logits, val_y)
          acc = utils.accuracy(logits, val_y)
          aves['vl'].update(loss.item())
          aves['va'].update(acc[0])

    for k, avg in aves.items():
      aves[k] = avg.item()
      trlog[k].append(aves[k])

    t_epoch = utils.time_str(timer_epoch.end())
    t_elapsed = utils.time_str(timer_elapsed.end())
    t_estimate = utils.time_str(timer_elapsed.end() / 
      (epoch - start_epoch + 1) * (config['n_epochs'] - start_epoch + 1))

    # formats output
    log_str = '[{}/{}] train {:.4f}(C)|{:.2f}'.format(
      str(epoch + start_epoch_from), str(config['n_epochs'] + start_epoch_from), aves['tl'], aves['ta'])
    writer.add_scalars('loss', {'train': aves['tl']}, epoch + start_epoch_from)
    writer.add_scalars('acc', {'train': aves['ta']}, epoch + start_epoch_from)
    if eval_val:
      log_str += ', val {:.4f}(C)|{:.2f}'.format(aves['vl'], aves['va'])
      writer.add_scalars('loss', {'val': aves['vl']}, epoch + start_epoch_from)
      writer.add_scalars('acc', {'val': aves['va']}, epoch + start_epoch_from)

    log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str)

    # saves model and meta-data
    if config.get('_parallel'):
      model_ = model.module
    else:
      model_ = model

    ckpt = {
      'file': __file__,
      'config': config,
      'epoch': epoch,
      'max_va': max(max_va, aves['va']),

      'encoder': ckpt['encoder'],
      'encoder_args': ckpt['encoder_args'],
      'encoder_state_dict': model_.enc.state_dict(),

      'classifier': config['classifier'],
      'classifier_args': config['classifier_args'],
      'classifier_state_dict': model_.head.state_dict(),

      'optimizer': config['optimizer'],
      'optimizer_args': config['optimizer_args'],
      'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
    torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))
    if aves['va'] > max_va:
      max_va = aves['va']
      torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))
    if config.get('save_epoch') and epoch % config['save_epoch'] == 0:
      torch.save(ckpt, os.path.join(ckpt_path, 'epoch-{}.pth'.format(epoch + start_epoch_from)))

    writer.flush()


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
  
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if args.n_batch_train:
    config['train_set_args']['n_batch'] = int(args.n_batch_train)
  if args.n_shot:
    config['train_set_args']['n_shot'] = int(args.n_shot)
  if args.sample_per_task:
    config['train_set_args']['n_query'] = int(args.sample_per_task/config['train_set_args']['n_way'] - args.n_shot)
  if args.path:
    config['path'] = "./save/task_samples/make_up_paper/{}".format(args.path)
    utils.log("load model from path: {}".format(config['path']))
    
  utils.log('{}y{}s_{}m_{}M'.format(
    config['train_set_args']['n_way'], 
    config['train_set_args']['n_shot'], 
    (config['train_set_args']['n_shot'] + config['train_set_args']['n_query']) * config['train_set_args']['n_way'],
    config['train_set_args']['n_batch']
    ))
  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  # utils.set_gpu(args.gpu)
  main(config)
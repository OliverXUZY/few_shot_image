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

import clip


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a vision encoder on vision language model")
    parser.add_argument(
        '--config', help='configuration file'
    )
    parser.add_argument(
        "--do_train", default=False, help="decide whether to fintune the model", action='store_true'
    )
    parser.add_argument(
        "--do_val", default=False, help="decide whether to validate the model after finetune", action='store_true'
    )
    parser.add_argument(
        "--do_test", default=False, help="decide whether to test the model on test set", action='store_true'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args.do_train, args.do_test)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    SEED = config.get('seed') or 0
    utils.log("seed: {}".format(SEED))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    ##### Dataset #####

    if args.do_test:
        test_set = datasets.make(config['dataset'], **config['test_set_args'])
        utils.log('test dataset: {} (x{}), {}'.format(
            test_set[0][0].shape, len(test_set), test_set.n_class), filename='test.txt')

        E = test_set.n_episode
        Y = test_set.n_way
        Q = test_set.n_query

        # query-set labels
        y = torch.arange(Y)[:, None]
        y = y.repeat(E, Q).flatten()
        y = y.cuda()                # [E * Y * Q]

        test_loader = DataLoader(test_set, E, num_workers=1, pin_memory=True)
    
    print("done test dataset loader")
    
    ##### Model #####
    if config.get('path'):
        assert os.path.exists(os.path.join(config['path'], config['ckpt']))
        ckpt = torch.load(os.path.join(config['path'], config['ckpt']))
        enc = encoders.load(ckpt)
    else:
        config['encoder_args'] = config.get('encoder_args') or dict()
        enc = encoders.make(config['encoder'], **config['encoder_args'])
        ckpt = {
        'encoder': config['encoder'],
        'encoder_args':  config['encoder_args'],
        }

    config['classifier_args'] = config.get('classifier_args') or dict()
    config['classifier_args']['in_dim'] = enc.get_out_dim()
    clf = classifiers.make(config['classifier'], **config['classifier_args'])

    if args.do_train:
        ckpt_name = '{}_{}_{}_{}y_{}m_{}M'.format(
            config['dataset'], ckpt['encoder'], config['classifier'],
            config['train_set_args']['n_way'], 
            (config['train_set_args']['n_shot'] + config['train_set_args']['n_query']) * config['train_set_args']['n_way'],
            config['train_set_args']['n_batch']
        )
    
    if args.do_test:
        ckpt_name = "vl_zero_shot"
    ckpt_path = os.path.join('./save/VL', ckpt_name)
    if not os.path.isdir(ckpt_path):
        utils.ensure_path(ckpt_path)
    utils.set_log_path(ckpt_path)
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

    print("done test model")

    ##### Evaluation #####
    if args.do_test:
        utils.log('{}_{}y{}q:'.format(config['classifier'], 
        config['test_set_args']['n_way'], config['test_set_args']['n_query']), filename='test.txt')
    
    
    enc.eval()
    aves_keys = ['va']
    aves = {k: utils.AverageMeter() for k in aves_keys}
    va_lst = []

    for epoch in range(1, config['n_epochs'] + 1):
        np.random.seed(epoch)

        with torch.no_grad():
            for (query, label, shot_names) in tqdm(test_loader, desc='test', leave=False):
                
                ### encode image
                query = query.cuda(non_blocking=True)   # [E,Y*Q(0,0,0,..,1,1,1...,...), C,H,W] [4,75, 3, 224,224]
                E, YQ = query.shape[:-3]
                query = query.flatten(0,-4)             # [E*Y*Q, C, H ,W] [300, 3, 224,224]
                q = enc(query)                          # [E*Y*Q, D] [300, 512] # in ViT-B32 D = 512
                q = q.view(1, E, YQ, -1)                # [QV = 1, E, Y * Q, D]

                ### encode text
                token_ep = list(map(
                    lambda x: enc.model.encode_text(
                        torch.concat(
                            [clip.tokenize(f"a photo of a {name}") for name in x]   # each element is [1,77] tokens for one sentence
                            ).cuda(non_blocking=True)                               # 4 eps for tokens for first of Y class [E,77] [4, 77]
                        ),        # 4 eps for text features for one of Y class  [E,D] [4, 512]
                    shot_names
                    ))            # list with length Y, each element is above [E,D]

                textfea_ep = torch.stack(token_ep)      # [Y,E,D] [5, 4, 512]
                textfea_ep = textfea_ep.transpose(0, 1) # [E,Y,D] [4, 5, 512]
                E,Y,D = textfea_ep.shape
                s = textfea_ep.view(1,E,Y,1,1,D)        # [SV = 1, E, Y, S, 1, D] to align with shape in clf

                logits = clf(s, q)                      # [1, E, Y*Q, Y]   [1, 4, 75, 5]
                
                logits = logits.flatten(0, -2)                  # [E * Y * Q, Y]
                acc = utils.accuracy(logits, y)
                aves['va'].update(acc[0])
                va_lst.append(acc[0].item())

            utils.log('[{}/{}]: acc={:.2f} +- {:.2f} (%)'.format(
            epoch, str(config['n_epochs']), aves['va'].item(), 
            utils.mean_confidence_interval(va_lst)), filename='test.txt')

if __name__ == "__main__":
    main()


    

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

templates = ['a photo of a {}',
             'itap of a {}.',
            'a bad photo of the {}.',
            'a origami {}.',
            'a photo of the large {}.',
            'a {} in a video game.',
            'art of the {}.',
            'a photo of the small {}.']

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
            lambda x: clip_model.encode_text(
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
    print("train: {}, val: {}, test: {}".format(args.do_train, args.do_val, args.do_test))
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    SEED = config.get('seed') or 0
    utils.log("seed: {}".format(SEED))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    ##### Dataset #####
    if args.do_train:
        train_set = datasets.make(config['dataset'], **config['train_set_args'])
        utils.log('meta-train dataset: split-{} {} (x{}), {}'.format(config['train_set_args']['split'],
            train_set[0][0].shape, len(train_set), train_set.n_class))
        
        TE = train_set.n_episode
        TY = train_set.n_way
        TQ = train_set.n_query

        # query-set labels
        train_y = torch.arange(TY)[:, None]
        train_y = train_y.repeat(TE, TQ).flatten()      # [TE * SV * TY * TQ]
        train_y = train_y.cuda()

        train_loader = DataLoader(train_set, TE, num_workers=1, pin_memory=True)

        utils.log("done train dataset loader")

    # meta-val
    eval_val = False
    if args.do_val:
        eval_val = True
        val_set = datasets.make(config['dataset'], **config['val_set_args'])
        utils.log('meta-val dataset: {} (x{}), {}'.format(
        val_set[0][0].shape, len(val_set), val_set.n_class))

        E = val_set.n_episode
        Y = val_set.n_way
        Q = val_set.n_query

        # query-set labels
        val_y = torch.arange(Y)[:, None]
        val_y = val_y.repeat(E, Q).flatten()  # [E * Y * Q]
        val_y = val_y.cuda()

        val_loader = DataLoader(val_set, E, num_workers=1, pin_memory=True)

        utils.log("done val dataset loader")
    

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
    
        utils.log("done test dataset loader")
    
    ##### Model #####
    if config.get('path'):
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

    ##### Optimizer and ckpt #####
    ckpt_name = config.get('ckpt_name') or None
    if args.do_train:
        ckpt_name = '{}_{}_{}_{}y_{}M'.format(
            config['dataset'], ckpt['encoder'], config['classifier'],
            config['train_set_args']['n_way'], 
            config['train_set_args']['n_batch']
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in enc.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config['optimizer_args']['weight_decay'],
            },
            {
                "params": [p for n, p in enc.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optimizers.make(config['optimizer'], optimizer_grouped_parameters, 
                              **config['optimizer_args'])
        start_epoch = 1
        max_va = 0.
        utils.log('num params: {}'.format(utils.count_params(enc)))

        timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()

        utils.log("done train model")
    
    
    if args.do_test:
        if not ckpt_name: # zero shot model
            ckpt_name = "vl_zero_shot"

        utils.log("done test model")


    save_path = config.get('save_path') or './save/VL'
    ## if config has 'path', it will overwrite all ckpt_path/ckpt_name
    ckpt_path = config.get('path') or os.path.join(save_path, ckpt_name)
    if not os.path.isdir(ckpt_path):
        utils.ensure_path(ckpt_path)
    utils.set_log_path(ckpt_path)
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

    ##### Testing #####
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
                    # ## shot_names: list with length Y, each element of list is a tuple with E names
                    # # [('vase', 'Alaskan Malamute', 'electric guitar', 'mixing bowl'),
                    # # ('red king crab', 'scoreboard', 'Dalmatian', 'Golden Retriever'),
                    # # ('trifle', 'lion', 'vase', 'red king crab'),
                    # # ('black-footed ferret', 'crate', 'nematode', 'front curtain'),
                    # # ('crate', 'Golden Retriever', 'bookstore', 'Dalmatian')]
                    # s = []
                    # for template in templates:
                    #     token_ep = list(map(
                    #         lambda x: enc.model.encode_text(
                    #             torch.concat(
                    #                 [clip.tokenize(template.format(name)) for name in x]   # each element is [1,77] tokens for one sentence
                    #                 ).cuda(non_blocking=True)                               # 4 eps ('vase', 'Alaskan Malamute', 'electric guitar', 'mixing bowl') for tokens for one of Y class [E,77] [4, 77]
                    #             ),        # 4 eps for text features for one of Y class  [E,D] [4, 512]
                    #         shot_names
                    #         ))            # list with length Y, each element is above [E,D]

                    #     textfea_ep = torch.stack(token_ep)      # [Y,E,D] [5, 4, 512]
                    #     s.append(textfea_ep)
                    # s = torch.stack(s)                      # [T=8,Y,E,D] 8 templates        [8, 5, 4, 512]
                    # s = s.transpose(0,2)                    # [E,Y,T,D]      [4, 5, 8, 512]
                    s = text_encoder(shot_names, enc.model)
                    # utils.log(s.shape)
                    s = s.unsqueeze(3).unsqueeze(0)         # [SV = 1, E, Y, S, V = 1, D]  [1, 4, 5, 8, 1, 512])

                    logits = clf(s, q)                      # [1, E, Y*Q, Y]   [1, 4, 75, 5]
                    
                    logits = logits.flatten(0, -2)                  # [E * Y * Q, Y]
                    acc = utils.accuracy(logits, y)
                    aves['va'].update(acc[0])
                    va_lst.append(acc[0].item())

                utils.log('[{}/{}]: acc={:.2f} +- {:.2f} (%)'.format(
                epoch, str(config['n_epochs']), aves['va'].item(), 
                utils.mean_confidence_interval(va_lst)), filename='test.txt')

    ##### Training and evaluation #####
    if args.do_train:
        utils.log("start training")
  
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

            enc.train() 
            ## change BatchNorm:
            if "RN" in ckpt['encoder']:
                enc.apply(utils.set_bn_eval)
            np.random.seed(epoch + SEED)

            # adjust learning rate
            lr = utils.decay_lr(epoch, config['n_epochs'], **config['optimizer_args'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            for (query, label, shot_names) in tqdm(train_loader, desc='train', leave=False):
                ### encode image
                query = query.cuda(non_blocking=True)   # [E,Y*Q(0,0,0,..,1,1,1...,...), C,H,W] [4,75, 3, 224,224]
                E, YQ = query.shape[:-3]
                query = query.flatten(0,-4)             # [E*Y*Q, C, H ,W] [300, 3, 224,224]
                q = enc(query)                          # [E*Y*Q, D] [300, 512] # in ViT-B32 D = 512
                q = q.view(1, E, YQ, -1)                # [QV = 1, E, Y * Q, D]

                ### encode text, does not update text encoder
                with torch.no_grad():
                    s = text_encoder(shot_names, enc.model)
                    s = s.unsqueeze(3).unsqueeze(0)         # [SV = 1, E, Y, S, V = 1, D]  [1, 4, 5, 8, 1, 512])


                logits = clf(s, q)                      # [1, E, Y*Q, Y]   [1, 4, 75, 5]
                
                logits = logits.flatten(0, -2)                  # [E * Y * Q, Y]

                loss = xent_loss(logits, train_y)
                acc = utils.accuracy(logits, train_y)
                aves['tl'].update(loss.item())
                aves['ta'].update(acc[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    
            # meta-val
            if eval_val:
                enc.eval()
                np.random.seed(SEED)
                
                with torch.no_grad():
                    for (query, label, shot_names) in tqdm(val_loader, desc='test', leave=False):
                        
                        ### encode image
                        query = query.cuda(non_blocking=True)   # [E,Y*Q(0,0,0,..,1,1,1...,...), C,H,W] [4,75, 3, 224,224]
                        E, YQ = query.shape[:-3]
                        query = query.flatten(0,-4)             # [E*Y*Q, C, H ,W] [300, 3, 224,224]
                        q = enc(query)                          # [E*Y*Q, D] [300, 512] # in ViT-B32 D = 512
                        q = q.view(1, E, YQ, -1)                # [QV = 1, E, Y * Q, D]

                        ### encode text
                        s = text_encoder(shot_names, enc.model)
                        s = s.unsqueeze(3).unsqueeze(0)         # [SV = 1, E, Y, S, V = 1, D]  [1, 4, 5, 8, 1, 512])

                        logits = clf(s, q)                      # [1, E, Y*Q, Y]   [1, 4, 75, 5]
                        
                        logits = logits.flatten(0, -2)                  # [E * Y * Q, Y]
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

            ckpt = {
            'file': __file__,
            'config': config,
            'epoch': epoch,
            'max_va': max(max_va, aves['va']),

            'encoder': ckpt['encoder'],
            'encoder_args': ckpt['encoder_args'],
            'encoder_state_dict': enc.state_dict(),

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

    
if __name__ == "__main__":
    main()


    

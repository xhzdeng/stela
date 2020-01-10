from __future__ import print_function

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from datasets.voc_dataset import VOCDataset
from datasets.collater import Collater
from models.stela import STELA
from utils.timer import Timer
from eval import evaluate

_t = Timer()


def train_model(args):
    #
    ds = VOCDataset(root_dir=args.train_dir)
    print('Number of Training Images is: {}'.format(len(ds)))
    scales = args.training_size + 32 * np.array([x for x in range(-5, 6)])
    collater = Collater(scales=scales, keep_ratio=False, multiple=32)
    loader = data.DataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=collater,
        shuffle=True,
        drop_last=True
    )
    #
    model = STELA(backbone=args.backbone, num_classes=2)
    if os.path.exists(args.pretrained):
        model.load_state_dict(torch.load(args.pretrained))
        print('Load pretrained model from {}.'.format(args.pretrained))
    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    iters_per_epoch = np.floor((len(ds) / float(args.batch_size)))
    num_epochs = int(np.ceil(args.max_iter / iters_per_epoch))
    iter_idx = 0
    for _ in range(num_epochs):
        for _, batch in enumerate(loader):
            iter_idx += 1
            if iter_idx > args.max_iter:
                break
            _t.tic()
            scheduler.step(epoch=iter_idx)
            model.train()

            if args.freeze_bn:
                if torch.cuda.device_count() > 1:
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            optimizer.zero_grad()
            ims, gt_boxes = batch['image'], batch['boxes']
            if torch.cuda.is_available():
                ims, gt_boxes = ims.cuda(), gt_boxes.cuda()
            losses = model(ims, gt_boxes)
            loss_cls, loss_reg = losses['loss_cls'].mean(), losses['loss_reg'].mean()
            if losses.__contains__('loss_ref'):
                loss_ref = losses['loss_ref'].mean()
                loss = loss_cls + (loss_reg + loss_ref) * 0.5
            else:
                loss = loss_cls + loss_reg
            if bool(loss == 0):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            if iter_idx % args.display == 0:
                info = 'iter: [{}/{}], time: {:1.3f}'.format(iter_idx, args.max_iter, _t.toc())
                if losses.__contains__('loss_ref'):
                    info = info + ', ref: {:1.3f}'.format(loss_ref.item())
                info = info + ', cls: {:1.3f}, reg: {:1.3f}'.format(loss_cls.item(), loss_reg.item())
                print(info)
            #
            if (arg.eval_iter > 0) and (iter_idx % arg.eval_iter) == 0:
                model.eval()
                if torch.cuda.device_count() > 1:
                    evaluate(model.module, args)
                else:
                    evaluate(model, args)
    #
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), './weights/deploy.pth')
    else:
        torch.save(model.state_dict(), './weights/deploy.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    # network
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--pretrained', type=str, default='')
    # dataset
    parser.add_argument('--train_dir', type=str, default='/path/to/yours')
    # training
    parser.add_argument('--training_size', type=int, default=640)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_iter', type=int, default=35000)
    parser.add_argument('--step_size', type=int, default=20000)
    parser.add_argument('--display', type=int, default=200)
    # testing
    parser.add_argument('--eval_iter', type=int, default=1000)
    parser.add_argument('--target_size', type=int, default=[800])
    parser.add_argument('--test_dir', type=str, default='/path/to/yours')
    parser.add_argument('--eval_dir', type=str, default='./eval/')
    parser.add_argument('--dataset', type=str, default='ICDAR 2015')
    #
    arg = parser.parse_args()
    train_model(arg)

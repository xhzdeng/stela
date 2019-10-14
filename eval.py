from __future__ import print_function

import os
import cv2
import torch
import codecs
import zipfile
import shutil
import argparse

from models.stela import STELA
from utils.detect import im_detect
from utils.bbox import rbox_2_aabb, rbox_2_quad
from utils.utils import sort_corners, is_image
from utils.timer import Timer

_t = Timer()


def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    # pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            # arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, filename)
    zipf.close()


def evaluate(model, args):
    #
    if args.dataset == 'ICDAR 2013':
        ims_dir = os.path.join(args.test_dir, args.dataset, 'Task 1/Test/IMS')
        eval_dir = os.path.join(args.eval_dir, 'icdar13')
    elif args.dataset == 'ICDAR 2015':
        ims_dir = os.path.join(args.test_dir, args.dataset, 'Task 1/Test/IMS')
        eval_dir = os.path.join(args.eval_dir, 'icdar15')
    elif args.dataset == 'ICDAR 2017':
        ims_dir = os.path.join(args.test_dir, args.dataset, 'Task 1/Test/IMS')
        eval_dir = os.path.join(args.eval_dir, 'icdar17')
    elif args.dataset == 'COCO':
        ims_dir = os.path.join(args.test_dir, args.dataset, 'Test/IMS')
        eval_dir = os.path.join(args.eval_dir, 'coco')
    else:
        raise NotImplementedError
    #
    out_dir = './temp'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #
    ims_list = [x for x in os.listdir(ims_dir) if is_image(x)]
    for idx, im_name in enumerate(ims_list):
        im_path = os.path.join(ims_dir, im_name)
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        _t.tic()
        dets = im_detect(model, im, target_sizes=args.target_size)
        print('\rim_detect: {:d}/{:d}, time: {:.3f}, '.format(idx + 1, len(ims_list), _t.toc()), end='')
        if args.dataset == 'ICDAR 2017':
            out_file = os.path.join(out_dir, im_name[:im_name.rindex('.')] + '.txt')
            out_file = out_file.replace('ts', 'res')
        elif args.dataset == 'COCO':
            im_index = im_name[im_name.rindex('_')+1:im_name.rindex('.')]
            out_file = os.path.join(out_dir, 'res_' + str(int(im_index)) + '.txt')
        else:
            out_file = os.path.join(out_dir, 'res_' + im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                continue
            if args.dataset == 'ICDAR 2013':
                res = rbox_2_aabb(dets[:, 2:])
                for k in range(dets.shape[0]):
                    f.write('{:.0f},{:.0f},{:.0f},{:.0f}\n'.format(
                        res[k, 0], res[k, 1], res[k, 2], res[k, 3])
                    )
            elif args.dataset == 'ICDAR 2015':
                res = sort_corners(rbox_2_quad(dets[:, 2:]))
                for k in range(dets.shape[0]):
                    f.write('{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}\n'.format(
                        res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                        res[k, 4], res[k, 5], res[k, 6], res[k, 7])
                    )
            elif args.dataset == 'ICDAR 2017':
                res = sort_corners(rbox_2_quad(dets[:, 2:]))
                for k in range(dets.shape[0]):
                    f.write('{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.2f}\n'
                            .format(res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                                    res[k, 4], res[k, 5], res[k, 6], res[k, 7],
                                    dets[k, 1])
                            )
            if args.dataset == 'COCO':
                res = rbox_2_aabb(dets[:, 2:])
                for k in range(dets.shape[0]):
                    f.write('{:.0f},{:.0f},{:.0f},{:.0f},{:.2f}\n'.format(
                        res[k, 0], res[k, 1], res[k, 2], res[k, 3], dets[k, 1])
                    )
    #
    zip_name = 'submit.zip'
    make_zip(out_dir, zip_name)
    shutil.move(os.path.join('./', zip_name), os.path.join(eval_dir, zip_name))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    if args.dataset == 'ICDAR 2013' or args.dataset == 'ICDAR 2015':
        os.system('cd {0} && python2 script.py -g=gt.zip -s=submit.zip '.format(eval_dir))
        print()
    else:
        # evaluated online
        raise NotImplementedError


def do_eval(args):
    model = STELA(backbone=args.backbone, num_classes=2)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    evaluate(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', dest='backbone', default='res50', type=str)
    parser.add_argument('--weights', type=str, default='./weights/deploy.pth')
    parser.add_argument('--target_size', dest='target_size', default=[800], type=int)
    parser.add_argument('--test_dir', nargs='?', type=str, default='/path/to/yours')
    parser.add_argument('--eval_dir', nargs='?', type=str, default='./eval/')
    parser.add_argument('--dataset', nargs='?', type=str, default='ICDAR 2013')
    arg = parser.parse_args()
    do_eval(arg)

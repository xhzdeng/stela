from __future__ import print_function

import os
import cv2
import torch
import argparse
import numpy as np

from models.stela import STELA
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import is_image, draw_caption



def demo(args):
    #
    model = STELA(backbone=args.backbone, num_classes=2)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    ims_list = [x for x in os.listdir(args.ims_dir) if is_image(x)]

    for _, im_name in enumerate(ims_list):
        im_path = os.path.join(args.ims_dir, im_name)
        src = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        cls_dets = im_detect(model, im, target_sizes=args.target_size)
        for j in range(len(cls_dets)):
            cls, scores = cls_dets[j, 0], cls_dets[j, 1]
            bbox = cls_dets[j, 2:]
            if len(bbox) == 4:
                draw_caption(src, bbox, '{:1.3f}'.format(scores))
                cv2.rectangle(src, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
            else:
                pts = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
                cv2.drawContours(src, pts, 0, color=(0, 255, 0), thickness=2)
                # display original anchors
                # if len(bbox) > 5:
                #     pts = np.array([rbox_2_quad(bbox[5:]).reshape((4, 2))], dtype=np.int32)
                #     cv2.drawContours(src, pts, 0, color=(0, 0, 255), thickness=2)
        # resize for better shown
        im = cv2.resize(src, (800, 800), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Detection Results', im)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--weights', type=str, default='./weights/deploy.pth')
    parser.add_argument('--ims_dir', type=str, default='/path/to/yours')
    parser.add_argument('--target_size', type=int, default='800')
    demo(parser.parse_args())

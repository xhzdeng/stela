import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bbox import bbox_overlaps, min_area_square
from utils.box_coder import BoxCoder
from utils.overlaps.rbox_overlaps import rbox_overlaps


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classifications, anchors, annotations, iou_thresh=0.5):
        losses = []
        batch_size = classifications.shape[0]

        for j in range(batch_size):
            classification = classifications[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            if bbox_annotation.shape[0] == 0:
                losses.append(torch.tensor(0).float().cuda())
                continue
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            indicator = bbox_overlaps(
                min_area_square(anchors[j, :, :]),
                min_area_square(bbox_annotation[:, :-1])
            )
            overlaps = rbox_overlaps(
                anchors[j, :, :].cpu().numpy(),
                bbox_annotation[:, :-1].cpu().numpy(),
                indicator.cpu().numpy(),
                thresh=1e-1
            )
            if not torch.is_tensor(overlaps):
                overlaps = torch.from_numpy(overlaps).cuda()
            iou_max, iou_argmax = torch.max(overlaps, dim=1)
            targets = (torch.ones(classification.shape) * -1).cuda()
            targets[torch.lt(iou_max, 0.4), :] = 0
            positive_indices = torch.ge(iou_max, iou_thresh)
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[iou_argmax, :]
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1
            alpha_factor = torch.ones(targets.shape).cuda() * self.alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            bin_cross_entropy = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bin_cross_entropy
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
        return torch.stack(losses).mean(dim=0, keepdim=True)


def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = torch.abs(inputs - targets)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    if size_average:
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(inputs,
                     targets,
                     beta=1. / 9,
                     alpha=0.5,
                     gamma=1.5,
                     size_average=True):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """
    assert beta > 0
    assert inputs.size() == targets.size() and targets.numel() > 0

    diff = torch.abs(inputs - targets)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    if size_average:
        return loss.mean()
    return loss.sum()


class RegressLoss(nn.Module):
    def __init__(self, func='smooth'):
        super(RegressLoss, self).__init__()
        self.box_coder = BoxCoder()
        if func == 'smooth':
            self.criteron = smooth_l1_loss
        elif func == 'mse':
            self.criteron = F.mse_loss
        elif func == 'balanced':
            self.criteron = balanced_l1_loss

    def forward(self, regressions, anchors, annotations, iou_thresh=0.5):
        losses = []
        batch_size = regressions.shape[0]
        for j in range(batch_size):
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            if bbox_annotation.shape[0] == 0:
                losses.append(torch.tensor(0).float().cuda())
                continue
            indicator = bbox_overlaps(
                min_area_square(anchors[j, :, :]),
                min_area_square(bbox_annotation[:, :-1])
            )
            overlaps = rbox_overlaps(
                anchors[j, :, :].cpu().numpy(),
                bbox_annotation[:, :-1].cpu().numpy(),
                indicator.cpu().numpy(),
                thresh=1e-1
            )
            if not torch.is_tensor(overlaps):
                overlaps = torch.from_numpy(overlaps).cuda()
            iou_max, iou_argmax = torch.max(overlaps, dim=1)
            positive_indices = torch.ge(iou_max, iou_thresh)
            assigned_annotations = bbox_annotation[iou_argmax, :]
            if positive_indices.sum() > 0:
                all_rois = anchors[j, positive_indices, :]
                gt_boxes = assigned_annotations[positive_indices, :]
                targets = self.box_coder.encode(all_rois, gt_boxes)
                loss = self.criteron(regression[positive_indices, :], targets)
                losses.append(loss)
            else:
                losses.append(torch.tensor(0).float().cuda())
        return torch.stack(losses).mean(dim=0, keepdim=True)



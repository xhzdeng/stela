import torch
import torch.nn as nn
import torchvision.models as models

from models.fpn import FPN, LastLevelP6P7
from models.heads import CLSHead, REGHead
from models.anchors import Anchors
from models.losses import FocalLoss, RegressLoss

from utils.nms_wrapper import nms
from utils.box_coder import BoxCoder
from utils.bbox import clip_boxes


class STELA(nn.Module):
    def __init__(self,
                 backbone='res50',
                 num_classes=2,
                 num_refining=1):
        super(STELA, self).__init__()
        self.anchor_generator = Anchors()
        self.num_anchors = self.anchor_generator.num_anchors
        self.init_backbone(backbone)
        self.fpn = FPN(
            in_channels_list=self.fpn_in_channels,
            out_channels=256,
            top_blocks=LastLevelP6P7(self.fpn_in_channels[-1], 256)
        )
        self.cls_head = CLSHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=1,
            num_anchors=self.num_anchors,
            num_classes=num_classes
        )
        self.reg_head = REGHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=1,
            num_anchors=self.num_anchors,
            num_regress=5
        )
        self.num_refining = num_refining
        if self.num_refining > 0:
            self.ref_heads = nn.ModuleList(
                [REGHead(
                    in_channels=256,
                    feat_channels=256,
                    num_stacked=1,
                    num_anchors=self.num_anchors,
                    num_regress=5
                ) for _ in range(self.num_refining)]
            )
            self.loss_ref = RegressLoss(func='smooth')
        self.loss_cls = FocalLoss()
        self.loss_reg = RegressLoss(func='smooth')
        self.box_coder = BoxCoder()

    def init_backbone(self, backbone):
        if backbone == 'res34':
            self.backbone = models.resnet34(pretrained=True)
            self.fpn_in_channels = [128, 256, 512]
        elif backbone == 'res50':
            self.backbone = models.resnet50(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
        elif backbone == 'resnext50':
            self.backbone = models.resnext50_32x4d(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
        else:
            raise NotImplementedError
        del self.backbone.avgpool
        del self.backbone.fc

    def ims_2_features(self, ims):
        c1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(ims)))
        c2 = self.backbone.layer1(self.backbone.maxpool(c1))
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return [c3, c4, c5]

    def forward(self, ims, gt_boxes=None):
        anchors_list, offsets_list = [], []
        original_anchors = self.anchor_generator(ims)
        anchors_list.append(original_anchors)
        features = self.fpn(self.ims_2_features(ims))

        # anchor refining
        if self.num_refining > 0:
            for i in range(self.num_refining):
                bbox_pred = torch.cat([self.ref_heads[i](feature) for feature in features], dim=1)
                refined_anchors = self.box_coder.decode(anchors_list[-1], bbox_pred, mode='wht').detach()
                anchors_list.append(refined_anchors)
                offsets_list.append(bbox_pred)

        cls_score = torch.cat([self.cls_head(x) for x in features], dim=1)
        bbox_pred = torch.cat([self.reg_head(x) for x in features], dim=1)
        if self.training:
            losses = dict()
            if self.num_refining > 0:
                ref_losses = []
                for i in range(self.num_refining):
                    ref_losses.append(self.loss_ref(
                        offsets_list[i], anchors_list[i],
                        gt_boxes, iou_thresh=(0.3 + i * 0.1))
                    )
                losses['loss_ref'] = torch.stack(ref_losses).mean(dim=0, keepdim=True)
            losses['loss_cls'] = self.loss_cls(cls_score, anchors_list[-1], gt_boxes, iou_thresh=0.5)
            losses['loss_reg'] = self.loss_reg(bbox_pred, anchors_list[-1], gt_boxes, iou_thresh=0.5)
            return losses
        else:
            return self.decoder(ims, anchors_list[-1], cls_score, bbox_pred)

    def decoder(self, ims, anchors, cls_score, bbox_pred, thresh=0.3, nms_thresh=0.3):
        bboxes = self.box_coder.decode(anchors, bbox_pred, mode='xywht')
        bboxes = clip_boxes(bboxes, ims)
        scores = torch.max(cls_score, dim=2, keepdim=True)[0]
        keep = (scores >= thresh)[0, :, 0]
        if keep.sum() == 0:
            return [torch.zeros(1), torch.zeros(1), torch.zeros(1, 5)]
        scores = scores[:, keep, :]
        anchors = anchors[:, keep, :]
        cls_score = cls_score[:, keep, :]
        bboxes = bboxes[:, keep, :]
        anchors_nms_idx = nms(torch.cat([bboxes, scores], dim=2)[0, :, :], nms_thresh)
        nms_scores, nms_class = cls_score[0, anchors_nms_idx, :].max(dim=1)
        output_boxes = torch.cat([
            bboxes[0, anchors_nms_idx, :],
            anchors[0, anchors_nms_idx, :]],
            dim=1
        )
        return [nms_scores, nms_class, output_boxes]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == '__main__':
    pass

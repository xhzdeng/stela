import numpy as np
import cv2
import torch

"""
aabb: [xmin, ymin, xmax, ymax]
rbox: [xmin, ymin, xmax, ymax, theta]
quad: [x1, y1, x2, y2, x3, y3, x4, y4]
"""


def bbox_overlaps(boxes, query_boxes):
    area = (query_boxes[:, 2] - query_boxes[:, 0]) * \
           (query_boxes[:, 3] - query_boxes[:, 1])
    iw = torch.min(torch.unsqueeze(boxes[:, 2], dim=1), query_boxes[:, 2]) - \
         torch.max(torch.unsqueeze(boxes[:, 0], 1), query_boxes[:, 0])
    ih = torch.min(torch.unsqueeze(boxes[:, 3], dim=1), query_boxes[:, 3]) - \
         torch.max(torch.unsqueeze(boxes[:, 1], 1), query_boxes[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    return intersection / ua


def rbox_overlaps(boxes, query_boxes, indicator=None, thresh=1e-1):
    # rewrited by cython
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    a_tt = boxes[:, 4]
    a_ws = boxes[:, 2] - boxes[:, 0]
    a_hs = boxes[:, 3] - boxes[:, 1]
    a_xx = boxes[:, 0] + a_ws * 0.5
    a_yy = boxes[:, 1] + a_hs * 0.5

    b_tt = query_boxes[:, 4]
    b_ws = query_boxes[:, 2] - query_boxes[:, 0]
    b_hs = query_boxes[:, 3] - query_boxes[:, 1]
    b_xx = query_boxes[:, 0] + b_ws * 0.5
    b_yy = query_boxes[:, 1] + b_hs * 0.5

    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = b_ws[k] * b_hs[k]
        for n in range(N):
            if indicator is not None and indicator[n, k] < thresh:
                continue
            ua = a_ws[n] * a_hs[n] + box_area
            rtn, contours = cv2.rotatedRectangleIntersection(
                ((a_xx[n], a_yy[n]), (a_ws[n], a_hs[n]), a_tt[n]),
                ((b_xx[k], b_yy[k]), (b_ws[k], b_hs[k]), b_tt[k])
            )
            if rtn == 1:
                ia = cv2.contourArea(contours)
                overlaps[n, k] = ia / (ua - ia)
            elif rtn == 2:
                ia = np.minimum(ua - box_area, box_area)
                overlaps[n, k] = ia / (ua - ia)
    return overlaps


def quad_2_rbox(quads):
    # http://fromwiz.com/share/s/34GeEW1RFx7x2iIM0z1ZXVvc2yLl5t2fTkEg2ZVhJR2n50xg
    if len(quads.shape) == 1:
        quads = quads[np.newaxis, :]
    rboxs = np.zeros((quads.shape[0], 5), dtype=np.float32)
    for i, quad in enumerate(quads):
        rbox = cv2.minAreaRect(quad.reshape([4, 2]))
        x, y, w, h, t = rbox[0][0], rbox[0][1], rbox[1][0], rbox[1][1], rbox[2]
        if np.abs(t) < 45.0:
            rboxs[i, :] = np.array([x, y, w, h, t])
        elif np.abs(t) > 45.0:
            rboxs[i, :] = np.array([x, y, h, w, 90.0 + t])
        else:
            if w > h:
                rboxs[i, :] = np.array([x, y, w, h, -45.0])
            else:
                rboxs[i, :] = np.array([x, y, h, w, 45])
    # (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    rboxs[:, 0:2] = rboxs[:, 0:2] - rboxs[:, 2:4] * 0.5
    rboxs[:, 2:4] = rboxs[:, 0:2] + rboxs[:, 2:4]
    rboxs[:, 0:4] = rboxs[:, 0:4].astype(np.int32)
    return rboxs


def rbox_2_quad(rboxs):
    if len(rboxs.shape) == 1:
        rboxs = rboxs[np.newaxis, :]
    if rboxs.shape[0] == 0:
        return rboxs
    quads = np.zeros((rboxs.shape[0], 8), dtype=np.float32)
    for i, rbox in enumerate(rboxs):
        w = rbox[2] - rbox[0]
        h = rbox[3] - rbox[1]
        x = rbox[0] + 0.5 * w
        y = rbox[1] + 0.5 * h
        theta = rbox[4]
        quads[i, :] = cv2.boxPoints(((x, y), (w, h), theta)).reshape((1, 8))

    return quads


def quad_2_aabb(quads):
    aabb = np.zeros((quads.shape[0], 4), dtype=np.float32)
    aabb[:, 0] = np.min(quads[:, 0::2], 1)
    aabb[:, 1] = np.min(quads[:, 1::2], 1)
    aabb[:, 2] = np.max(quads[:, 0::2], 1)
    aabb[:, 3] = np.max(quads[:, 1::2], 1)
    return aabb


def rbox_2_aabb(rboxs):
    if len(rboxs.shape) == 1:
        rboxs = rboxs[np.newaxis, :]
    if rboxs.shape[0] == 0:
        return rboxs
    quads = rbox_2_quad(rboxs)
    aabbs = quad_2_aabb(quads)
    return aabbs


def min_area_square(rboxs):
    w = rboxs[:, 2] - rboxs[:, 0]
    h = rboxs[:, 3] - rboxs[:, 1]
    ctr_x = rboxs[:, 0] + w * 0.5
    ctr_y = rboxs[:, 1] + h * 0.5
    s = torch.max(w, h)
    return torch.stack((
        ctr_x - s * 0.5, ctr_y - s * 0.5,
        ctr_x + s * 0.5, ctr_y + s * 0.5),
        dim=1
    )


def clip_boxes(boxes, ims):
    _, _, h, w = ims.shape
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=w)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=h)
    return boxes


if __name__ == '__main__':
    pass

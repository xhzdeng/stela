import cv2
import numpy as np
import numpy.random as npr
import torch
import torchvision.transforms as transforms


def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j*2] = corners[idx*2]
            sorted[i, j*2+1] = corners[idx*2+1]
    return sorted


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


def is_image(filename):
    return any(filename.endswith(ext) for ext in [".bmp", ".png", ".jpg", ".jpeg", ".JPG"])


def rescale(im, target_size, max_size, keep_ratio, multiple=32):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if keep_ratio:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im_scale_x = np.floor(im.shape[1] * im_scale / multiple) * multiple / im.shape[1]
        im_scale_y = np.floor(im.shape[0] * im_scale / multiple) * multiple / im.shape[0]
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
    else:
        target_size = int(np.floor(float(target_size) / multiple) * multiple)
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
    return im, im_scale


class Rescale(object):
    def __init__(self, target_size=600, max_size=2000, keep_ratio=True):
        self._target_size = target_size
        self._max_size = max_size
        self._keep_ratio = keep_ratio

    def __call__(self, im):
        if isinstance(self._target_size, list):
            random_scale_inds = npr.randint(0, high=len(self._target_size))
            target_size = self._target_size[random_scale_inds]
        else:
            target_size = self._target_size
        im, im_scales = rescale(im, target_size, self._max_size, self._keep_ratio)
        return im, im_scales


class Normailize(object):
    def __init__(self):
        # RGB: https://github.com/pytorch/vision/issues/223
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, im):
        im = self._transform(im)
        return im


class Reshape(object):
    def __init__(self, unsqueeze=True):
        self._unsqueeze = unsqueeze
        return

    def __call__(self, ims):
        if not torch.is_tensor(ims):
            ims = torch.from_numpy(ims.transpose((2, 0, 1)))
        if self._unsqueeze:
            ims = ims.unsqueeze(0)
        return ims


if __name__ == '__main__':
    pass
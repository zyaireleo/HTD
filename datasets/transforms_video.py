"""
Transforms and data augmentation for sequence level images, bboxes and masks.
"""
import random

import PIL
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from numpy import random as rand

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate

try:
    import accimage
except ImportError:
    accimage = None
from collections.abc import Sequence
import numbers
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Check(object):
    def __init__(self, ):
        pass

    def __call__(self, img, depth, target):
        fields = ["labels"]
        if "boxes" in target:
            fields.append("boxes")
        if "masks" in target:
            fields.append("masks")

        ### check if box or mask still exist after transforms
        if "boxes" in target or "masks" in target:
            if "boxes" in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            if False in keep:
                for k in range(len(keep)):
                    if not keep[k] and "boxes" in target:
                        target['boxes'][k] = target['boxes'][k] // 1000.0  # [0, 0, 0, 0]

        target['valid'] = keep.to(torch.int32)

        return img, depth, target

def crop(clip, depths, target, region):
    cropped_image, cropped_depths = [], []
    for _, (image, depth) in enumerate(zip(clip, depths)):
        cropped_image.append(F.crop(image, *region))
        cropped_depths.append(F.crop(depth, *region))

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    return cropped_image, cropped_depths, target


def hflip(clip, depths, target):
    flipped_image, flipped_depth = [], []
    for image in clip:
        flipped_image.append(F.hflip(image))
    for depth in depths:
        flipped_depth.append(F.hflip(depth))

    w, h = clip[0].size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, flipped_depth, target

def resize(clip, depths, target, size, max_size=None, binarize=True):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(clip[0].size, size, max_size)
    rescaled_image, rescaled_depth = [], []
    for _, (image, depth) in enumerate(zip(clip, depths)):
        rescaled_image.append(F.resize(image, size))
        rescaled_depth.append(F.resize(depth, size))

    if target is None:
        return rescaled_image, rescaled_depth, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image[0].size, clip[0].size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        if target['masks'].shape[0] > 0:
            if binarize:
                target['masks'] = interpolate(
                    target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5
            else:
                # TODO: interpolate for each object
                target['masks'] = interpolate(
                    target['masks'][:, None].float(), size, mode="nearest")[:, 0]  # > 0.5
        else:
            target['masks'] = torch.zeros((target['masks'].shape[0], h, w))
    return rescaled_image, rescaled_depth, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, flows, target):
        # todo
        print('using RandCrop???<-----transforms_video--line257')
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, flows, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, depth, target: dict):
        w = random.randint(self.min_size, min(img[0].width, self.max_size))
        h = random.randint(self.min_size, min(img[0].height, self.max_size))
        region = T.RandomCrop.get_params(img[0], [h, w])
        return crop(img, depth, target, region)


class RandomContrast(object):
    def __init__(self, lower=0.8, upper=1.2):  # lower=0.5, upper=1.5
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, depth, target):
        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            image *= alpha
        return image, depth, target


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, depth, target):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            image += delta
            image = np.clip(image, 0, 255)
        return image, depth, target


class RandomSaturation(object):
    def __init__(self, lower=0.8, upper=1.2):  # lower=0.5, upper=1.5
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, depth, target):
        if rand.randint(2):
            image[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, depth, target


# TODO: change p from 0.5 to 0.2
class RandomLightingNoise(object):
    def __init__(self, p=0.2):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        self.p = p

    def __call__(self, image, depth, target):
        if random.random() < self.p:
            # if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
            depth = shuffle(depth)
        return image, depth, target


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, depth, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
            depth = cv2.cvtColor(depth, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, depth, target


class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            # RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, clip, depths, target):
        imgs, ds = [], []
        for _, (img, depth) in enumerate(zip(clip, depths)):
            img = np.asarray(img).astype('float32')
            depth = np.asarray(depth).astype('float32')
            img, depth, target = self.rand_brightness(img, depth, target)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])  # order change
            else:
                distort = Compose(self.pd[1:])
            img, depth, target = distort(img, depth, target)
            img, depth, target = self.rand_light_noise(img, depth, target)  # shuffle channel
            imgs.append(Image.fromarray(img.astype('uint8')))
            ds.append(Image.fromarray(depth.astype('uint8')))
        return imgs, ds, target

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            # NOTE: caption for 'left' and 'right' should also change
            caption = target['caption']
            target['caption'] = caption.replace('left', '@').replace('right', 'left').replace('@', 'right')
            return hflip(img, depth, target)
        return img, depth, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None, binarize=True):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.binarize = binarize

    def __call__(self, img, depth, target=None):
        size = random.choice(self.sizes)
        return resize(img, depth, target, size, self.max_size, self.binarize)



class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, depth, target):
        if random.random() < self.p:
            return self.transforms1(img, depth, target)
        return self.transforms2(img, depth, target)


class ToTensor(object):
    def __call__(self, clip, depths, target):
        img, ds = [], []
        for im in clip:
            img.append(F.to_tensor(im))
        for depth in depths:
            ds.append(F.to_tensor(depth))
        return img, ds, target


class Normalize(object):
    def __init__(self, mean, std, depth_mean, depth_std):
        self.mean = mean
        self.std = std
        self.flow_mean = depth_mean
        self.flow_std = depth_std

    def __call__(self, clip, depths, target=None):
        image, ds = [], []
        for im in clip:
            image.append(F.normalize(im, mean=self.mean, std=self.std))
        for depth in depths:
            ds.append(F.normalize(depth, mean=self.flow_mean, std=self.flow_std))
        if target is None:
            return image, ds, None
        target = target.copy()
        h, w = image[0].shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, ds, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, depth, target):
        for t in self.transforms:
            image, depth, target = t(image, depth, target)
        return image, depth, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

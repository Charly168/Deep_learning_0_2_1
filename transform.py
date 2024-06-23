import random
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import torch

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, label):
        image = F.to_tensor(image)
        # label = F.to_tensor(label)
        return image, label


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image.flip(-1)  # 水平翻转图片
        return image, target

class RandomVerticalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        if random.random() < self.prob:
            image = image.flip(0)  # 水平翻转图片
        return image, label

class Resize(object):

    "Resize 图像"
    def __init__(self,size,interpolation=None):
        self.size = size
        self.interpolation = interpolation
    def __call__(self,image,label):
        if self.interpolation == 'bilinear':self.interpolation = InterpolationMode.BILINEAR
        elif self.interpolation == 'bicubic':self.interpolation = InterpolationMode.BILINEAR
        elif self.interpolation == 'nearst': self.interpolation = InterpolationMode.NEAREST
        else: self.interpolation = InterpolationMode.BILINEAR
        image = F.resize(image, self.size, interpolation=self.interpolation)
        return image,label

class Normalize(object):
    def __init__(self,image_mean=None,image_std=None):
        if image_mean is None:
            self.image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            self.image_std = [0.229, 0.224, 0.225]
        else:
            self.image_mean = image_mean
            self.image_std = image_std
    def __call__(self, image,label):
        """标准化处理"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None], label
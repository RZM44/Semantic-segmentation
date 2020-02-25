import torch 
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    convert image from [0,255] to [0.0,1.0]
    image = (image - mean)/std
    The target not normalize
    Args:
        mean(tuple) : means for each channel.
        std(tuple) : standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = np.array(image).astype(np.float32)
        target = np.array(target).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std

        return image, target

class ToTensor(object):
    """Converts a PIL Image or numpy.ndarray (H x W x C) 
    to a torch.FloatTensor of shape (C x H x W) 
    """
    def __call__(self, image, target):
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        target = np.array(target).astype(np.float32)

        image = torch.from_numpy(image).float()
        target = torch.from_numpy(target).float()
        return image, target

class Compose(object):
    """The Compose need rewrite to carry two inputs
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image,target)
        return image, target

class RandomHorizontalFlip(object):
    """ Random Horizontal Flip the image and target p = 0.5
    """
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self,image,target):
        if random.random() < self.p:
            return F.hflip(image), F.hflip(target)
        return image, target

class RandomScale(object):
    """ Scale the given PIL Image at a random size in given range 
    """
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, image, target):
        assert image.size == target.size
        scale = random.uniform(self.scale_range[0],self.scale_range[1])                             ## first version 
        target_size = (int(image.size[1]*scale),int(image.size[0]*scale))
        return F.resize(image, target_size, Image.BILINEAR), F.resize(target, target_size, Image.NEAREST)

class RandomCrop(object):
    """ Crop the given PIL Image at a random location with given size
    """
    def __init__(self, size):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

    @staticmethod    
    def get_params(image, output_size):
        w, h = image.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        return i, j, th, tw
    def __call__(self, image, target):
        assert image.size == target.size
        if image.size[0] < self.size[1]:
            image = F.pad(image, padding=int((1 + self.size[1] - image.size[0]) / 2))
            target = F.pad(target, padding=int((1 + self.size[1] - target.size[0]) / 2))

        # pad the height if needed
        if image.size[1] < self.size[0]:
            image = F.pad(image, padding=int((1 + self.size[0] - image.size[1]) / 2))
            target = F.pad(target, padding=int((1 + self.size[0] - target.size[1]) / 2)) 
        i, j, h , w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(target, i, j, h, w)

class FixScale(object):
    """ Fixscale the given PIL Image 
    """
    def __init__(self, scale_target_size):
        self.target_size = scale_target_size

    def __call__(self, image, target):
        return F.resize(image, self.target_size, Image.BILINEAR), F.resize(target, self.target_size, Image.NEAREST)

class CenterCrop(object):
    """ Crop the given PIL Image at center  with given size
    """
    def __init__(self, size):
        if isinstance(size, tuple):
             self.size = size
        else:
             self.size = (size, size)

    def __call__(self, image, target):
        return F.center_crop(image, self.size), F.center_crop(target, self.size)



import torch 
import random
import numpy as np

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
        print(type(image))
        print(type(target))
        return image, target


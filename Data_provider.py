from __future__ import print_function
import os
import numpy as np
from PIL import Image
import torch.utils.data as data
class VOCSegmentation(data.Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
       'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, crop_size=None):
        self.root = os.path.expanduser(root)
        _voc_root = os.path.join(self.root, 'VOC_trainval_aug')
        _train_dir = os.path.join(_voc_root, 'JPEGImages')
        _val_dir = os.path.join(_voc_root,'SementationClassAug')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.crop_size = crop_size

        if download:
            self.download()

        if self.train:
            _list_f = os.path.join(_voc_root, 'trainaug.txt')
        else:
            _list_f = os.path.join(_voc_root, 'val.txt')
        self.images = []
        self.lables = []
        with open(_list_f, 'r') as lines:
            for line in lines:
                image = os.path.join(_train_dir, line + ".jpg")
                lable = os.path.join(_val_dir,line + ".png")
                assert os.path.isfile(image)
                assert os.path.isfile(lable)
                self.images.append(image)
                self.lables.append(lable)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.lables[index])
        """
        img, target = preprocess(img, target,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
        """
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


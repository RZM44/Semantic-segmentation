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
        _voc_root = self.root
        _train_dir = os.path.join(_voc_root, 'VOCdevkit/VOC2012/JPEGImages/')
        _val_dir = os.path.join(_voc_root,'SegmentationClassAug')
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
                image = os.path.join(_train_dir, line.rstrip('\n') + ".jpg")
                lable = os.path.join(_val_dir,line.rstrip('\n') + ".png")
                assert os.path.isfile(image),"no image in:" + image
                assert os.path.isfile(lable),"no label in:" + lable
                self.images.append(image)
                self.lables.append(lable)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.lables[index])
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    
    transform_train = trainsforms.Compse([
          ])
    voc_train = VOCSegmentation(root='./data',train=True)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=2)

    for ii, img, target in enumerate(dataloader):
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img)
            plt.subplot(212)
            plt.imshow(target)
            break
    plt.show(block=True)

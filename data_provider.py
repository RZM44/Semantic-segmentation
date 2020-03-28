from __future__ import print_function
import os
import numpy as np
from PIL import Image
import torch.utils.data as data
from utils import transforms as trans
class VOCSegmentation(data.Dataset):
    """Dataset class for PASCAL VOC 2012
       totally 21 classes(including backgroud)
    """
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
       'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor'
    ]

    def __init__(self, root, set_name, transform=None, download=False, crop_size=None):
        self.root = os.path.expanduser(root)
        _voc_root = self.root
        _train_dir = os.path.join(_voc_root, 'VOCdevkit/VOC2012/JPEGImages/')
        _label_dir = os.path.join(_voc_root,'SegmentationClassAug')
        self.transform = transform
        self.set_name = set_name
        self.crop_size = crop_size
        if download:
            self.download()

        if self.set_name is 'train':
            _list_f = os.path.join(_voc_root, 'trainaug.txt')
        elif self.set_name is 'val':
            _list_f = os.path.join(_voc_root, 'newval.txt')
        elif self.set_name is 'oldval':
            _list_f = os.path.join(_voc_root, 'oldval.txt')
        elif self.set_name is 'oldtrain':
            _list_f = os.path.join(_voc_root, 'train.txt')
        else:
            _list_f = os.path.join(_voc_root, 'test.txt')
        self.datas = []
        self.lables = []
        with open(_list_f, 'r') as lines:
            for line in lines:
                data = os.path.join(_train_dir, line.rstrip('\n') + ".jpg")
                lable = os.path.join(_label_dir, line.rstrip('\n') + ".png")
                assert os.path.isfile(data),"no data in:" + data
                assert os.path.isfile(lable),"no label in:" + lable
                self.datas.append(data)
                self.lables.append(lable)

    def __getitem__(self, index):
        image = Image.open(self.datas[index]).convert('RGB')
        target = Image.open(self.lables[index])
        
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target

    def __len__(self):
        return len(self.datas)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from utils import transforms as trans
    import numpy as np
    np.random.seed(32)
    transform_train = trans.Compose([
          #trans.RandomCrop(256),
          trans.FixRangeScale(np.random.randint(64,256)),
          #trans.RandomCrop(256),
          trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          trans.ToTensor(),
          ])

    """ 
    transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ])
    """
    transform_val = trans.Compose([
          trans.FixScale((256,256)),
          trans.CenterCrop(256),
          trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          trans.ToTensor(),
          ])
    
    voc_train = VOCSegmentation(root='./data',set_name='val',transform=transform_train)
    #np.random.seed(32)
    #voc_val = VOCSegmentation(root='./data',train=False,transform=transform_val)
    #def worker_init_fn(worker_id):                                                          
        #np.random.seed(np.random.get_state()[1][0] + worker_id)
    dataloader = DataLoader(voc_train, batch_size=2, shuffle=True, num_workers=0) #,worker_init_fn=worker_init_fn)
    #dataloader = DataLoader(voc_val, batch_size=3, shuffle=True, num_workers=0)
    #print(len(dataloader))
    #np.random.seed(32)
    for (img, tag) in dataloader:
        image = img[0]
        target = tag[0]
        image = image.numpy()
        target = target.numpy().astype(np.uint8)
        image = image.transpose((1,2,0))
        image *= (0.229, 0.224, 0.225)
        image += (0.485, 0.456, 0.406)
        image *= 255.0
        image = image.astype(np.uint8)
        plt.figure()
        plt.title('display')
        plt.subplot(231)
        plt.imshow(image)
        plt.subplot(232)
        plt.imshow(target)
        plt.subplot(233)
        plt.imshow(image)
        plt.imshow(target,alpha=0.5)
        break
    for (img, tag) in dataloader:
        image = img[0]
        target = tag[0]
        image = image.numpy()
        target = target.numpy().astype(np.uint8)
        image = image.transpose((1,2,0))
        image *= (0.229, 0.224, 0.225)
        image += (0.485, 0.456, 0.406)
        image *= 255.0
        image = image.astype(np.uint8)
        plt.figure()
        plt.title('display')
        plt.subplot(234)
        plt.imshow(image)
        plt.subplot(235)
        plt.imshow(target)
        plt.subplot(236)
        plt.imshow(image)
        plt.imshow(target,alpha=0.5)
        break
    plt.show(block=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import transforms as trans

def get_pascal_colour():
 CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
           'tv/monitor'
    ]
 return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], 
                   [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], 
                   [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                   [0, 64, 128]]) 

def decode_segmap(label):
    label_colours = get_pascal_colour()
    num_class = len(label_colours)
    r = label.copy()
    g = label.copy()
    b = label.copy()
    for i in range(0, num_class):
        r[label==i] = label_colours[i, 0]
        g[label==i] = label_colours[i, 1]
        b[label==i] = label_colours[i, 2]
    rgb = np.zeros((label.shape[0], label.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

 

import numpy as np
import torch

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        with np.errstate(invalid='ignore'):
            Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        intersection = np.diag(self.confusion_matrix)
        gt_set = np.sum(self.confusion_matrix, axis=1)
        pre_set = np.sum(self.confusion_matrix, axis=0)
        union = gt_set + pre_set - intersection
        with np.errstate(invalid='ignore'):
            iou = intersection / union
        MIoU = np.nanmean(iou)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

if __name__ == '__main__':
    import torch
    size = 4
    x = torch.zeros([size, size], dtype=torch.int)
    x[:, 0] = 5 
    print("Predict:\n", x)
    eval = Evaluator(6)
    y = torch.zeros([size, size], dtype=torch.int64)
    y[:, 0] = 5
    y[0, :] = 5
    print("Target\n", y)
    eval.add_batch(y.numpy(), x.numpy())
    print(eval.confusion_matrix)
    Acc = eval.Pixel_Accuracy_Class()
    Miou = eval.Mean_Intersection_over_Union()
    print("Acc: {}, MIOU: {}".format(Acc, Miou))
    eval.reset()
    print(eval.confusion_matrix)



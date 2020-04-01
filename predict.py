import numpy as np
import torch
from data_provider import VOCSegmentation
from experiment_builder import ExperimentBuilder
from utils.arg_extractor import get_args
import utils.transforms as trans
from model.deeplab import DeepLab
import matplotlib.pyplot as plt
from tools import prediction
from utils.metrics import Evaluator
args = get_args()
rng = np.random.RandomState(seed=args.seed)

torch.manual_seed(seed=args.seed)

transform_train = trans.Compose([
          trans.RandomHorizontalFlip(),
          #trans.FixScale((args.crop_size,args.crop_size)),
          trans.RandomScale((0.5,2.0)),
          #trans.FixScale(args.crop_size),
          trans.RandomCrop(args.crop_size),
          trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          trans.ToTensor(),
          ])

transform_val = trans.Compose([
          #trans.FixScale((args.crop_size,args.crop_size)),
          trans.FixScale(args.crop_size),
          trans.CenterCrop(args.crop_size),
          trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          trans.ToTensor(),
          ])
if(args.aug == True):
    voc_train = VOCSegmentation(root='./data', set_name='train', transform=transform_train)
else:
    voc_train = VOCSegmentation(root='./data', set_name='oldtrain', transform=transform_train)
if(args.newval == True):
    voc_val = VOCSegmentation(root='./data', set_name='val', transform=transform_val)
else:
    voc_val =VOCSegmentation(root='./data', set_name='oldval', transform=transform_val)
voc_test = VOCSegmentation(root='./data', set_name='test', transform=transform_val)
#train_data = torch.utils.data.DataLoader(voc_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
#val_data = torch.utils.data.DataLoader(voc_val, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
#test_data = torch.utils.data.DataLoader(voc_test, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
train_data = torch.utils.data.DataLoader(voc_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_data = torch.utils.data.DataLoader(voc_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_data = torch.utils.data.DataLoader(voc_test, batch_size=args.batch_size, shuffle=False, num_workers=4)


custom_net = DeepLab(args.output_stride, args.num_class)

ss_experiment = ExperimentBuilder(network_model=custom_net, num_class=args.num_class, experiment_name=args.experiment_name, num_epochs=args.num_epochs, train_data=train_data, val_data=val_data, test_data=test_data, learn_rate=args.learn_rate, mementum=args.mementum, weight_decay=args.weight_decay, use_gpu=args.use_gpu, continue_from_epoch=args.continue_from_epoch)

plt.switch_backend('agg')

evaluator = Evaluator(args.num_class)
for idx, (img, tag) in enumerate(val_data):
    predicts= ss_experiment.run_predicted_iter(img, tag)
    for i in range(args.batch_size):
        image = img[i]
        target = tag[i]
        predict = predicts[i]
        image = image.numpy()
        target = target.numpy().astype(np.uint8)
        image = image.transpose((1,2,0))
        image *= (0.229, 0.224, 0.225)
        image += (0.485, 0.456, 0.406)
        image *= 255.0
        image = image.astype(np.uint8)
        evaluator.add_batch(target, predict)
        miou = evaluator.Mean_Intersection_over_Union()
        acc = evaluator.Pixel_Accuracy()
        target = prediction.decode_segmap(target)
        predict = prediction.decode_segmap(predict)
        fig = plt.figure()
        plt.title('{}_{}'.format("Miou: "+ str(miou), "Pixelacc: "+ str(acc)))
        plt.subplot(131)
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(target)
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(predict)
        #plt.imshow(target,alpha=0.5)
        plt.axis('off')
        fig.savefig('./image/'+ str(i) + '.png')
        plt.close(fig)
        print("image save success")
    break






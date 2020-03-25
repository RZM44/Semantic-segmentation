import numpy as np
import torch
from data_provider import VOCSegmentation
from experiment_builder import ExperimentBuilder
from utils.arg_extractor import get_args
import utils.transforms as trans
from model.deeplab import DeepLab

args = get_args()
rng = np.random.RandomState(seed=args.seed)

torch.manual_seed(seed=args.seed)

transform_train = trans.Compose([
          trans.RandomHorizontalFlip(),
          trans.RandomScale((0.5,2.0)),
          trans.RandomCrop(args.crop_size),
          trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          trans.ToTensor(),
          ])

transform_val = trans.Compose([
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


custom_net = DeepLab(args.output_stride, args.num_class,  args.sy_bn)

ss_experiment = ExperimentBuilder(network_model=custom_net, num_class=args.num_class, experiment_name=args.experiment_name, num_epochs=args.num_epochs, train_data=train_data, val_data=val_data, test_data=test_data, learn_rate=args.learn_rate, mementum=args.mementum, weight_decay=args.weight_decay, use_gpu=args.use_gpu, continue_from_epoch=args.continue_from_epoch)

current_epoch_losses = {"val_miou": [], "val_acc": [], "val_loss": []}
current_epoch_losses = ss_experiment.run_validation_epoch()
out_string = "_".join(
                 ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
print(out_string)






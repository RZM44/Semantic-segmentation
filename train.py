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
          trans.RandomScale((0.5,2.0)),
          trans.RandomCrop(513),
          trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          trans.ToTensor(),
          ])

transform_val = trans.Compose([
          trans.FixScale(513),
          trans.CenterCrop(513),
          trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          trans.ToTensor(),
          ])
voc_train = VOCSegmentation(root='./data', train=True, transform=transform_train)
voc_val = VOCSegmentation(root='./data', train=False, transform=transform_val)
train_data = torch.utils.data.DataLoader(voc_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_data = torch.utils.data.DataLoader(voc_val, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_data = torch.utils.data.DataLoader(voc_val, batch_size=args.batch_size, shuffle=False, num_workers=4)


custom_net = DeepLab(args.output_stride)

ss_experiment = ExperimentBuilder(network_model=custom_net, num_class=args.num_class, experiment_name=args.experiment_name, num_epochs=args.num_epochs, train_data=train_data, val_data=val_data, test_data=test_data, learn_rate=args.learn_rate, mementum=args.mementum, weight_decay=args.weight_decay, use_gpu=args.use_gpu, continue_from_epoch=args.continue_from_epoch)

experiment_metrics, test_metrics = ss_experiment.run_experiment()





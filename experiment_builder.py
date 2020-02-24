import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time

from utils.storage import save_statistics
from utils.loss import FacalLoss
from utils.scheduler import PolyLR
from utils.metrics import Evaluator

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, num_class, experiment_name, num_epochs, train_data, val_data, test_data, learn_rate, mementum, weight_decay, use_gpu, continue_from_epoch=-1):
    super(ExperimentBuilder, self).__init__()

    self.experiment_name = experiment_name
    self.model = network_model
    self.model.reset_parameters()
    self.nclass = num_class
    self.learn_rate = learn_rate
    self.mementum = mementum
    self.weight_decay = weight_decay
    self.device = torch.cuda.current_device()

    if torch.cuda.device_count() > 1 and use_gpu:
        self.device = torch.cuda.current_device()
        self.model.to(self.device)
        self.model = nn.DataParallel(module=self.model)
        print('Use Mutil GPU', self.device)
    elif torch.cuda.device_count() == 1 and use_gpu:
        self.device = torch.cuda.current_device()
        self.model.to(self.device)  
        print('Use GPU', self.device)
    else:
        self.device = torch.device('cpu')  
        print('Use CPU', self.device)

    self.train_data = train_data
    self.val_data = val_data
    self.test_data = test_data 
    self.num_epochs = num_epochs

    train_params = [{'params': model.get_backbone.parameters(), 'lr': self.learn_rate},
                    {'params': model.get_classifier_parameters(), 'lr': self.learn_rate * 10}]
    self.optimizer = torch.optim.SGD(train_params, momentum=self.mementum, weight_decay=self.weight_decay)
    self.criterion = FocalLoss(ignore_index=255, size_average=True).to(self.device)
    self.scheduler = PolyLR(self.optimizer, max_iters= self.num_epochs, power=0.9)
    self.evaluator = Evaluator(self.num_class)
 
    total_num_params = 0
    for param in self.parameters():
        total_num_params += np.prod(param.shape)
    print('System learnable parameters')

    self.experiment_folder = os.path.abspath(experiment_name)
    self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
    self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
    print(self.experiment_folder, self.experiment_logs)
    # Set best models to be at 0 since we are just starting
    self.best_val_model_idx = 0
    self.best_val_model_acc = 0.

    if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
        os.mkdir(self.experiment_folder)  # create the experiment directory

    if not os.path.exists(self.experiment_logs):
        os.mkdir(self.experiment_logs)  # create the experiment log directory

    if not os.path.exists(self.experiment_saved_models):
        os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory
    
    if continue_from_epoch == -2:
        try:
            self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx='latest')  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        except:
            print("Model objects cannot be found, initializing a new model and starting from scratch")
            self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()

    def run_train_iter(self, image, target):
        
        self.train()
        self.evaluator.reset()
        image = image.to(self.device)
        target = target.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model.forward(image)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        predicted = output.data.cpu().numpy()
        predicted = np.argmax(predicted, axis=1)

        self.evaluator.add_batch(target, predicted)
        miou = self.edvaluator.Mean_Intersection_over_Union()
        return loss.data.detach().cpu().numpy(), miou
    
    def run_evaluation_iter(self, image, target):
        
        self.eval()
        self.evaluator.reset()
        image = image.to(self.device)
        target = target.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model.forward(image)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        predicted = output.data.cpu().numpy()
        predicted = np.argmax(predicted, axis=1)

        self.evaluator.add_batch(target, predicted)
        miou = self.edvaluator.Mean_Intersection_over_Union()
        return loss.data.detach().cpu().numpy(), miou
    
    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        
        state['network'] = self.state_dict()
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
                    model_idx))))))

    def run_training_epoch(self, current_epoch_losses):
        with tqdm.tqdm(total=len(self.train_data), file=sys.stdout) as pbar_train:  # create a progress bar for training
            for idx, (image, target) in enumerate(self.train_data):  # get data batches
                loss, miou = self.run_train_iter(image, target)  # take a training iter step
                current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                current_epoch_losses["train_miou"].append(accuracy)  # add current iter acc to the train acc list
                pbar_train.update(1)
                pbar_train.set_description("loss: {:.4f}, miou: {:.4f}".format(loss, accuracy))

        return current_epoch_losses
    
    def run_validation_epoch(self, current_epoch_losses):
        with tqdm.tqdm(total=len(self.val_data), file=sys.stdout) as pbar_train:  # create a progress bar for training
            for idx, (x, y) in enumerate(self.train_data):  # get data batches
                loss, accuracy = self.run_train_iter(x=x, y=y)  # take a training iter step
                current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
                pbar_train.update(1)
                pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

        return current_epoch_losses

    def load_model(self, model_save_dir, model_save_name, model_idx):
        
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_acc'], state
    
    def run_experiment(self):

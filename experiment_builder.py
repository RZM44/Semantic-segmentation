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
from utils.loss import FocalLoss, CrossEntropyLoss
from utils.scheduler import PolyLR
from utils.metrics import Evaluator

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, num_class, experiment_name, num_epochs, train_data, val_data, test_data, learn_rate, mementum, weight_decay, use_gpu, continue_from_epoch=-1):
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.model = network_model
        #self.model.reset_parameters()
        self.num_class = num_class
        self.learn_rate = learn_rate
        self.mementum = mementum
        self.weight_decay = weight_decay

        if(self.experiment_name == "test"):
            print('Testmode')
        
        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            print('Use Mutil GPU', self.device)
            print('GPU number', torch.cuda.device_count())
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

        train_params = [{'params': network_model.get_backbone_params(), 'lr': self.learn_rate},
                        {'params': network_model.get_classifier_params(), 'lr': self.learn_rate * 1}]
        self.optimizer = torch.optim.SGD(train_params, momentum=self.mementum, weight_decay=self.weight_decay)
        #self.criterion = FocalLoss(ignore_index=255, size_average=True).to(self.device)
        self.criterion = CrossEntropyLoss(size_average=True, ignore_index=255).to(self.device)
        self.scheduler = PolyLR(self.optimizer, max_iters=self.num_epochs*len(self.train_data), power=0.9)
        self.evaluator = Evaluator(self.num_class)
     
        total_num_params = 0
        for param in network_model.parameters():
            total_num_params += np.prod(param.shape)
        print('System learnable parameters', total_num_params)
        num_conv_layers = 0
        for name, value in self.named_parameters():
            if all(item in name for item in ['conv', 'weight']):
                num_conv_layers += 1
        print('number of conv layers', num_conv_layers)
        

        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.

        if not os.path.exists(self.experiment_folder): 
            os.mkdir(self.experiment_folder) 

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models) 
        
        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  
                self.starting_epoch = self.state['current_epoch_idx'] + 1
                self.scheduler.step()
                self.scheduler.last_epoch = self.state['last_epoch']
                print("restart from epoch ",self.state['current_epoch_idx'])
                print("backbone learning rate: ", self.optimizer.param_groups[0]['lr'])
                print("classifier learning rate: ", self.optimizer.param_groups[1]['lr'])
                print("iterations: ", self.scheduler.last_epoch)
                print("base_lr:", self.scheduler.base_lrs)
                
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  
            self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  
            self.starting_epoch = self.state['current_epoch_idx'] + 1
            self.scheduler.step()
            self.scheduler.last_epoch = self.state['last_epoch']
            print("restart from epoch ",self.state['current_epoch_idx'])
            print("backbone learning rate: ", self.optimizer.param_groups[0]['lr'])
            print("classifier learning rate: ", self.optimizer.param_groups[1]['lr'])
            print("iterations: ", self.scheduler.last_epoch)
            print("base_lr:", self.scheduler.base_lrs)
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
        loss = self.criterion(output, target.long())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        predicted = output.data.cpu().numpy()
        target = target.cpu().numpy()
        predicted = np.argmax(predicted, axis=1)

        self.evaluator.add_batch(target, predicted)
        miou = self.evaluator.Mean_Intersection_over_Union()
        acc = self.evaluator.Pixel_Accuracy()
        return loss.data.detach().cpu().numpy(), miou, acc
    
    def run_evaluation_iter(self, image, target):
        
        self.eval()
        self.evaluator.reset()
        image = image.to(self.device)
        target = target.to(self.device)
        
        output = self.model.forward(image)
        loss = self.criterion(output, target.long())

        predicted = output.data.cpu().numpy()
        target = target.cpu().numpy()
        predicted = np.argmax(predicted, axis=1)

        self.evaluator.add_batch(target, predicted)
        miou = self.evaluator.Mean_Intersection_over_Union()
        acc = self.evaluator.Pixel_Accuracy()
        return loss.data.detach().cpu().numpy(), miou, acc
   
    def run_predicted_iter(self, image, target):

        self.eval()
        self.evaluator.reset()
        image = image.to(self.device)
        target = target.to(self.device)

        output = self.model.forward(image)
        loss = self.criterion(output, target.long())

        predicted = output.data.cpu().numpy()
        target = target.cpu().numpy()
        predicted = np.argmax(predicted, axis=1)

        self.evaluator.add_batch(target, predicted)
        miou = self.evaluator.Mean_Intersection_over_Union()
        acc = self.evaluator.Pixel_Accuracy()
        return loss.data.detach().cpu().numpy(), miou, acc, predicted, image, target

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        
        state['network'] = self.model.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        #state['scheduler'] = self.scheduler.state_dict()
        state['last_epoch'] = self.scheduler.last_epoch
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))

    def run_training_epoch(self, current_epoch_losses):
        with tqdm.tqdm(total=len(self.train_data), file=sys.stdout) as pbar_train:  
            for idx, (image, target) in enumerate(self.train_data): 
                loss, miou, acc = self.run_train_iter(image, target)  
                current_epoch_losses["train_loss"].append(loss)  
                current_epoch_losses["train_miou"].append(miou)
                current_epoch_losses["train_acc"].append(acc)  
                pbar_train.update(1)
                if(torch.cuda.device_count() >= 1):
                    m = torch.cuda.get_device_properties(0).total_memory/1e9
                    c = torch.cuda.max_memory_cached(0)/1e9
                    a = torch.cuda.max_memory_allocated(0)/1e9
                    pbar_train.set_description("Training: loss: {:.4f}, miou: {:.4f}, Pacc: {:.4f}, memory: {:.2f}GB, cached:{:.2f}GB, allocated:{:.2f}GB".format(loss, miou, acc, m, c, a))
                else:
                    pbar_train.set_description("Training: loss: {:.4f}, miou: {:.4f}, Pacc: {:.4f}".format(loss, miou, acc))
                    

        return current_epoch_losses
    
    def run_validation_epoch(self, current_epoch_losses):
        with tqdm.tqdm(total=len(self.val_data), file=sys.stdout) as pbar_val:  
            for idx, (image, target) in enumerate(self.val_data): 
                loss, miou, acc = self.run_evaluation_iter(image, target)  
                current_epoch_losses["val_loss"].append(loss) 
                current_epoch_losses["val_miou"].append(miou)
                current_epoch_losses["val_acc"].append(acc)  
                pbar_val.update(1)
                pbar_val.set_description("Validating: loss: {:.4f}, miou: {:.4f}, Pacc: {:.4f}".format(loss, miou, acc))

        return current_epoch_losses
    
    def run_testing_epoch(self, current_epoch_losses):
        with tqdm.tqdm(total=len(self.test_data), file=sys.stdout) as pbar_test:  
            for idx, (image, target) in enumerate(self.test_data): 
                loss, miou, acc = self.run_evaluation_iter(image, target)  
                current_epoch_losses["test_loss"].append(loss) 
                current_epoch_losses["test_miou"].append(miou)  
                current_epoch_losses["test_acc"].append(acc)
                pbar_test.update(1)
                pbar_test.set_description("Testing: loss: {:.4f}, miou: {:.4f}, Pacc: {:.4f}".format(loss, miou, acc))

        return current_epoch_losses

    def load_model(self, model_save_dir, model_save_name, model_idx):
        
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.model.load_state_dict(state_dict=state['network'])
        self.optimizer.load_state_dict(state_dict=state['optimizer'])
        #self.scheduler.load_state_dict(state_dict=state['scheduler'])
        self.scheduler.last_epoch = state['last_epoch']
        return state['best_val_model_idx'], state['best_val_model_acc'], state
    
    def run_experiment(self):
        total_losses = {"train_miou": [], "train_acc": [], "train_loss": [], "val_miou": [], "val_acc": [],
                        "val_loss": [], "curr_epoch": []}  
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_miou": [], "train_acc": [], "train_loss": [],"val_miou": [], "val_acc": [], "val_loss": []}

            current_epoch_losses = self.run_training_epoch(current_epoch_losses)
            #print(self.optimizer.param_groups[0]['lr'])
            current_epoch_losses = self.run_validation_epoch(current_epoch_losses)

            val_mean_miou = np.mean(current_epoch_losses['val_miou'])
            if val_mean_miou > self.best_val_model_acc:  
                self.best_val_model_acc = val_mean_miou  
                self.best_val_model_idx = epoch_idx  

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) 

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            epoch_elapsed_time = time.time() - epoch_start_time  
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx),"Iteration {}:".format(self.scheduler.last_epoch), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_acc'] = self.best_val_model_acc
            self.state['best_val_model_idx'] = self.best_val_model_idx
            if(self.experiment_name != "test"):
                if(epoch_idx==0 or (epoch_idx+1)%10==0):
                    self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
                self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model", model_idx='latest', state=self.state)
            
        if(self.experiment_name != "test"):
            print("Generating test set evaluation metrics")
            self.load_model(model_save_dir=self.experiment_saved_models, model_idx='latest',
                            model_save_name="train_model")
            current_epoch_losses = {"test_miou": [], "test_acc": [], "test_loss": []}  

            current_epoch_losses = self.run_testing_epoch(current_epoch_losses=current_epoch_losses)

            test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  

            save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)
        else:
            test_losses = 0
        return total_losses, test_losses 

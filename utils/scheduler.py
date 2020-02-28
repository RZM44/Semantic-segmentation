from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  
        self.min_lr = min_lr                
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
         return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs] 

if __name__ == '__main__':
    import torch
    train_params = [{'params': [torch.randn(1, requires_grad=True)], 'lr': 1 },{'params': [torch.randn(1, requires_grad=True)], 'lr': 2}]
    optimizer = torch.optim.SGD(train_params)
    scheduler = PolyLR(optimizer, max_iters=10, power=0.9)
    print("initial learing rate {1,2}, power = 0.9")
    for i in range(1, 10):
        scheduler.step()
        print('Iter {}, lr {}, lr {}'.format(i, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
        print(scheduler.base_lrs)


import torch
from pooraka.trainer import train_cifar, train_imagenet
from pooraka.inference import infer_cifar, infer_imagenet
from pooraka.dataset import get_cifar_queue
from pooraka.utils import save_checkpoint, CrossEntropyLabelSmooth
import torch.nn as nn
import logging

class learning:
    def __init__(self, args, model, save_model, logging):
        self.args = args
        self.model = model
        self.save_model = save_model
        self.logging = logging

        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                    self.args.learning_rate, 
                                    momentum=self.args.momentum, 
                                    weight_decay=self.args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    float(self.args.epochs),
                                                                    eta_min=0.0)
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.train_queue = None
        self.valid_queue = None

        self.best_acc1 = 0

    def load_data(self):
        if self.args.dataset.startswith('cifar'):
            self.train_queue, self.valid_queue = get_cifar_queue(self.args)

    def run(self):

        self.load_data()
        
        for epoch in range(self.args.epochs):
            if self.args.dataset.startswith('cifar'):
                train_acc, train_obj = train_cifar(self.args, 
                                                    self.train_queue, 
                                                    self.model, 
                                                    self.criterion, 
                                                    self.optimizer, 
                                                    logging_mode=self.logging)
                valid_acc, valid_obj = infer_cifar(self.args, 
                                                    self.valid_queue, 
                                                    self.model, 
                                                    self.criterion, 
                                                    logging_mode=self.logging)

                self.scheduler.step()

                is_best = valid_acc > self.best_acc1
                self.best_acc1 = max(valid_acc, self.best_acc1)

                if self.save_model:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'best_acc1': self.best_acc1,
                        'optimizer' : self.optimizer.state_dict(),
                    }, is_best, self.args.save)
            if self.logging:
                logging.info('best_acc1 %f', self.best_acc1)



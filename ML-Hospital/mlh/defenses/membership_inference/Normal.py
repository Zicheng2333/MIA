import torch
import numpy as np
import os
import time
from runx.logx import logx
import torch.nn.functional as F
from mlh.defenses.membership_inference.trainer import Trainer
import torch.nn as nn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TrainTargetNormal(Trainer):
    def __init__(self, model, device, num_class, epochs=100, learning_rate=0.01, momentum=0.9, weight_decay=5e-4, smooth_eps=0.8, log_path="./"):

        super().__init__()

        self.model = model
        self.device = device
        self.num_class = num_class
        self.epochs = epochs
        self.smooth_eps = smooth_eps

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), learning_rate, momentum, weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)

        self.criterion = nn.CrossEntropyLoss()

        self.log_path = log_path
        logx.initialize(logdir=self.log_path,
                        coolname=False, tensorboard=False)

    @staticmethod
    def _sample_weight_decay():
        # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
        weight_decay = np.logspace(-6, 5, num=45, base=10.0)
        weight_decay = np.random.choice(weight_decay)
        print("Sampled weight decay:", weight_decay)
        return weight_decay

    def eval(self, data_loader):

        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():

            for img, label in data_loader:
                img, label = img.to(self.device), label.to(self.device)
                logits = self.model.forward(img)

                predicted = torch.argmax(logits, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            final_acc = 100 * correct / total

        return final_acc

    def train(self, train_loader, test_loader):
        print("###################Start training###################")
        t_start = time.time()

        # check whether path exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        try:
            test_acc = self.eval(test_loader)
            print("Initial test acc:", test_acc)
            for e in range(1, self.epochs + 1):
                total_train_loss = 0
                batch_n = 0
                self.model.train()
                for img, label in train_loader:
                    self.model.zero_grad()
                    batch_n += 1

                    img, label = img.to(self.device), label.to(self.device)
                    # print("img", img.shape)
                    logits = self.model(img)
                    loss = self.criterion(logits, label)

                    total_train_loss+=loss.item()
                    loss.backward()
                    self.optimizer.step()

                trainLoss = f"Epoch {e},Train Loss: {total_train_loss/len(train_loader.dataset)}"
                logx.msg(trainLoss)

                train_acc = self.eval(train_loader)
                test_acc = self.eval(test_loader)

                logx.msg('Train Epoch: %d, Total Sample: %d, Train Acc: %.3f, Test Acc: %.3f, Total Time: %.3fs' % (
                    e, len(train_loader.dataset), train_acc, test_acc, time.time() - t_start))
                self.scheduler.step()

            torch.save(self.model.state_dict(), os.path.join(self.log_path, "%s.pth" % self.model_save_name))
        except KeyboardInterrupt:
            pass




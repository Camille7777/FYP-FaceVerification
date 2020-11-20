import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import tqdm
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model.cnn_model import CNN, ContrastiveLoss
from dataset.dataset import LfwDataset
import numpy as np

train_number_epochs = 20


class Trainer:
    def __init__(self, model, learning_rate, batch_size, use_cuda=True):
        self.model = model()
        self.device = torch.device('cuda:0') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        self.loss_fcn = ContrastiveLoss()
        self.optimizer = SGD(self.model.parameters(), lr=learning_rate)
        self.data_root = 'dataset/lfw_cropped/split_data/train'
        self.datasets = [LfwDataset(os.path.join(self.data_root, f'0{i}'))
                         for i in tqdm.tqdm(range(1, 11), desc='loading data...')]
        self.counter = []
        self.loss_history = []
        self.iteration_number = 0

    def get_10_folder_data(self, index):
        valid_dataset = self.datasets[index]
        train_dataset = sum([self.datasets[i] for i in range(10) if i != index])
        return valid_dataset, train_dataset

    '''def k_folder_iteration(self, num_epochs):
        train_loss_sum, valid_loss_sum = 0, 0
        train_acc_sum, valid_acc_sum = 0, 0

        for k in range(10):
            valid, train = self.get_10_folder_data(k)
            valid_loader = DataLoader(valid, batch_size=self.batch_size, shuffle=True)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            train_ls = self.train_k(train_loader, num_epochs)
            valid_ls = self.train_k(valid_loader, num_epochs)

            print('*' * 25, 'Cross', k + 1, '*' * 25)
            print('train_loss:%.6f' % train_ls[-1][0], 'train_acc:%.4f\n' % valid_ls[-1][1],
                  'valid loss:%.6f' % valid_ls[-1][0], 'valid_acc:%.4f' % valid_ls[-1][1])
            train_loss_sum += train_ls[-1][0]
            valid_loss_sum += valid_ls[-1][0]
            train_acc_sum += train_ls[-1][1]
            valid_acc_sum += valid_ls[-1][1]
        print('#' * 10, 'Final 10-cross validation result', '#' * 10)

        print('train_loss_sum:%.4f' % (train_loss_sum / 10), 'train_acc_sum:%.4f\n' % (train_acc_sum / 10),
              'valid_loss_sum:%.4f' % (valid_loss_sum / 10), 'valid_acc_sum:%.4f' % (valid_acc_sum / 10))'''

    '''
    def train_k(self, valid_dataset, train_dataset, train_num_epochs):
        train_ls, valid_ls = [], []
        self.model.cuda()
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(0, train_num_epochs):
            for i, data in enumerate(train_dataloader, 0):
                img1, img2, img1_soft_biometrics, img2_soft_biometrics, train_label = data
                img1, img2, img1_soft_biometrics, img2_soft_biometrics, train_label = img1.to(self.device), \
                                                                                      img2.to(self.device), \
                                                                                      img1_soft_biometrics.to(
                                                                                          self.device), \
                                                                                      img2_soft_biometrics.to(
                                                                                          self.device), \
                                                                                      train_label.to(
                                                                                          self.device)
                output1, output2 = self.model(img1, img2)
                loss_contrastive = self.loss_fcn(output1, output2, train_label)
                self.optimizer.zero_grad()
                loss_contrastive.backward()
                self.optimizer.step()
                train_ls.append(loss_contrastive.item())
                
        for i, data in enumerate(valid_dataloader, 0):
            img1, img2, img1_soft_biometrics, img2_soft_biometrics, train_label = data
            img1, img2, img1_soft_biometrics, img2_soft_biometrics, train_label = img1.to(self.device), \
                                                                                  img2.to(self.device), \
                                                                                  img1_soft_biometrics.to(self.device), \
                                                                                  img2_soft_biometrics.to(self.device), \
                                                                                  train_label.to(self.device)
            output1, output2 = self.model(img1, img2)
            loss_contrastive = self.loss_fcn(output1, output2, train_label)
            self.optimizer.zero_grad()
            loss_contrastive.backward()
            self.optimizer.step()
            valid_ls.append(loss_contrastive.item())
        return train_ls, valid_ls
    '''

    def train(self):
        self.model.cuda()
        dataset = sum([self.datasets[i] for i in range(10)])
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(0, train_number_epochs):
            counter = 0
            for i, data in enumerate(train_dataloader, 0):
                img1, img2, _, _, train_label = data
                img1, img2, train_label = img1.to(self.device), img2.to(self.device), train_label.to(self.device)
                counter += len(train_label)
                print("Pair {}\n".format(counter))
                self.model.train()
                output1, output2 = self.model(img1, img2)
                loss_contrastive = self.loss_fcn(output1, output2, train_label)
                print("loss before step: {}\n".format(loss_contrastive.item()))
                self.optimizer.zero_grad()
                loss_contrastive.backward()
                self.optimizer.step()
                self.model.eval()
                output1, output2 = self.model(img1, img2)
                loss_contrastive = self.loss_fcn(output1, output2, train_label)
                print("loss after step: {}\n".format(loss_contrastive.item()))
                predict = (F.pairwise_distance(output1, output2) < 1).float()
                print('Accurancy:', (predict == train_label).float().mean().item())

                if i % 10 == 0:
                    self.iteration_number += 10
                    self.counter.append(self.iteration_number)
                    self.loss_history.append(loss_contrastive.item())
            print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))
        show_plot(self.counter, self.loss_history)


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


t = Trainer(CNN, learning_rate=1e-2, batch_size=32, use_cuda=True)
t.train()

'''loss_func = nn.CrossEntropyLoss()
t = Trainer(CNN, learning_rate=1e-2, weight_decay=1e-1, batch_size=64, use_cuda=True)
t.k_folder_iteration(3)'''


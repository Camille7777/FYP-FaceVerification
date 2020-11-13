import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import os
import matplotlib.pyplot as plt
import torch.nn as nn

from model.cnn_model import CNN, ContrastiveLoss
from dataset.dataset import LfwDataset

class Trainer:
    def __init__(self, model, learning_rate=1e-2, weight_decay=1e-1, batch_size=64, use_cuda=True):
        self.model = model()
        self.device = torch.device('cuda:0') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        self.loss_fcn = ContrastiveLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.data_root = 'dataset/lfw_cropped/split_data/train'
        self.datasets = [LfwDataset(os.path.join(self.data_root, f'0{i}'))
                         for i in tqdm.tqdm(range(1, 11), desc='loading data...')]

    def get_10_folder_data(self, index):
        valid_dataset = self.datasets[index]
        train_dataset = sum([self.datasets[i] for i in range(10) if i != index])
        return valid_dataset, train_dataset

    def k_folder_iteration(self, num_epochs=3):
        train_loss_sum, valid_loss_sum = 0, 0
        train_acc_sum, valid_acc_sum = 0, 0

        for k in range(10):
            valid, train = self.get_10_folder_data(k)
            valid_loader = DataLoader(valid, batch_size=self.batch_size, shuffle=True)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)

            train_ls = self.train(train_loader, num_epochs)
            valid_ls = self.train(valid_loader, num_epochs)

            print('*' * 25, 'Cross', k + 1, '*' * 25)
            print('train_loss:%.6f' % train_ls[-1][0], 'train_acc:%.4f\n' % valid_ls[-1][1],
                  'valid loss:%.6f' % valid_ls[-1][0], 'valid_acc:%.4f' % valid_ls[-1][1])
            train_loss_sum += train_ls[-1][0]
            valid_loss_sum += valid_ls[-1][0]
            train_acc_sum += train_ls[-1][1]
            valid_acc_sum += valid_ls[-1][1]
        print('#' * 10, 'Final 10-cross validation result', '#' * 10)

        print('train_loss_sum:%.4f' % (train_loss_sum / 10), 'train_acc_sum:%.4f\n' % (train_acc_sum / 10),
              'valid_loss_sum:%.4f' % (valid_loss_sum / 10), 'valid_acc_sum:%.4f' % (valid_acc_sum / 10))

    def train(self, data_loader, num_epochs):
        train_ls, test_ls = [], []
        self.model().cuda()
        for epoch in range(0, num_epochs):
            for i, data in enumerate(data_loader, 0):
                img1, img2, img1_soft_biometrics, img2_soft_biometrics, train_label = data
                img1, img2, img1_soft_biometrics, img2_soft_biometrics, train_label = img1.to(self.device), img2.to(self.device), img1_soft_biometrics.to(self.device), img2_soft_biometrics.to(self.device), train_label.to(self.device)
                output1, output2 = self.model(img1, img2)
                loss_contrastive = self.loss_fcn(output1, output2, train_label)
                loss = loss_func(loss_contrastive, train_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # print(train_ls,test_ls)
        return train_ls, test_ls


loss_func = nn.CrossEntropyLoss()
t = Trainer(CNN, learning_rate=1e-2, batch_size=64, use_cuda=True)
t.k_folder_iteration(3)

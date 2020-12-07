import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
import tqdm
import os

from model.cnn_model import CNN, ContrastiveLoss
from dataset.dataset import LfwDataset

train_number_epochs = 1


class Trainer:
    def __init__(self, model, learning_rate=1e-2, batch_size=32, use_cuda=True):
        self.model = model
        self.device = torch.device('cuda:0') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        self.loss_fcn = ContrastiveLoss()
        self.optimizer = Adam
        self.lr = learning_rate
        self.data_root = 'dataset/lfw_cropped/split_data/train'
        self.datasets = [LfwDataset(os.path.join(self.data_root, f'0{i}'))
                         for i in tqdm.tqdm(range(1, 11), desc='loading data...')]
        self.counter = []
        self.loss_history = []
        self.iteration_number = 0
        self.loss_func = nn.CrossEntropyLoss()

    # separate for train dataset and valid dataset
    def get_10_fold_data(self, index):
        valid_dataset = self.datasets[index]
        train_dataset = sum([self.datasets[i] for i in range(10) if i != index])
        return valid_dataset, train_dataset

    def k_fold_iteration(self, num_epochs):
        train_loss_sum, valid_loss_sum = 0, 0
        train_acc_sum, valid_acc_sum = 0, 0

        for k in range(10):
            valid, train = self.get_10_fold_data(k)
            train_ls, valid_ls = self.train_k(valid, train, num_epochs, seed=k)

            print('*' * 25, 'Number ', k + 1, ' fold', '*' * 25)
            print('train_loss:%.4f' % train_ls[-1][0], 'train_acc:%.4f\n' % train_ls[-1][1],
                  'valid loss:%.4f' % valid_ls[-1][0], 'valid_acc:%.4f' % valid_ls[-1][1])
            train_loss_sum += train_ls[-1][0]
            valid_loss_sum += valid_ls[-1][0]
            train_acc_sum += train_ls[-1][1]
            valid_acc_sum += valid_ls[-1][1]
        print('#' * 10, 'Final 10-fold cross validation result', '#' * 10)

        print('train_loss_sum:%.4f' % (train_loss_sum / 10), 'train_acc_sum:%.4f\n' % (train_acc_sum / 10),
              'valid_loss_sum:%.4f' % (valid_loss_sum / 10), 'valid_acc_sum:%.4f' % (valid_acc_sum / 10))

    def train_k(self, valid_dataset, train_dataset, train_num_epochs, seed):
        train_ls, valid_ls = [], []
        torch.manual_seed(seed)
        model = self.model().to(self.device)
        optimizer = self.optimizer(model.parameters(), lr=self.lr, betas=(0.9, 0.98))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        model.train()
        for epoch in range(train_num_epochs):
            for i, data in tqdm.tqdm(enumerate(train_dataloader, 0),
                                     desc='Training Epoch_%s' % epoch,
                                     total=len(train_dataloader)):
                img1, img2,  _, _, train_label = data
                img1, img2, train_label = img1.to(self.device), img2.to(self.device), train_label.to(self.device)

                output1, output2 = model(img1, img2)
                loss_contrastive = self.loss_fcn(output1, output2, train_label)
                optimizer.zero_grad()
                loss_contrastive.backward()
                optimizer.step()
                train_dist = F.pairwise_distance(output1, output2)
                train_predict_lb = (train_dist < 1).int()
                train_accurancy = (train_predict_lb == train_label).float().mean().item()
                train_ls.append(self.res(loss_contrastive.item(), train_accurancy))
            count = 0
            with torch.no_grad():
                model.eval()
                for j, data1 in tqdm.tqdm(enumerate(valid_dataloader),
                                          desc='iteration on valid set...',
                                          total=len(valid_dataloader)):
                    img01, img02, _, _, valid_label = data1
                    img01, img02, valid_label = img01.to(self.device), img02.to(self.device), valid_label.to(self.device)
                    output01, output02 = model(img01, img02)
                    loss_contrastive = self.loss_fcn(output01, output02, valid_label)
                    valid_dist = F.pairwise_distance(output01, output02)
                    valid_predict_lb = (valid_dist < 1).int()
                    count += (valid_predict_lb == valid_label).float().sum()
                    valid_accurancy = count.item() / len(valid_dataloader.dataset)
                    valid_ls.append(self.res(loss_contrastive.item(), valid_accurancy))
        torch.save(model.state_dict(), f'model_state_dict/model_{seed}.pkl')
        return train_ls, valid_ls

    def res(self, x, y):
        return x, y


t = Trainer(CNN, learning_rate=1e-2, batch_size=32, use_cuda=True)
t.k_fold_iteration(1)

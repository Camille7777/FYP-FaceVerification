import torch
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
import tqdm
import os

from model.cnn_model import CNN, ContrastiveLoss
from dataset.dataset import LfwDataset
from train_fusion import *


class Trainer:
    def __init__(self, model, learning_rate=2e-4, batch_size=32, use_cuda=True):
        self.model = model
        self.device = torch.device('cuda:0') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        self.loss_fcn = ContrastiveLoss()
        self.optimizer = Adam
        self.lr = learning_rate
        self.data_root = 'dataset/lfw_cropped/split_data/test'
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

            print('*' * 25, 'Number ', k, ' fold', '*' * 25)
            train_ls, valid_ls = self.train_k(valid, train, num_epochs, seed=k, fusion_method='append', log_path='log/output.xlsx')
            train_loss_sum += train_ls[-1][0]
            valid_loss_sum += valid_ls[-1][0]
            train_acc_sum += train_ls[-1][1]
            valid_acc_sum += valid_ls[-1][1]
        print('#' * 10, 'Final 10-fold cross validation result', '#' * 10)

        print('train_loss_sum:%.6f' % (train_loss_sum / 10), 'train_acc_sum:%.6f\n' % (train_acc_sum / 10),
              'valid_loss_sum:%.6f' % (valid_loss_sum / 10), 'valid_acc_sum:%.6f' % (valid_acc_sum / 10))

    def train_k(self, valid_dataset, train_dataset, train_num_epochs, seed, fusion_method, log_path):
        '''if not os.path.isfile(log_path):
            df = pd.DataFrame(columns=["Fold number", "Training epoch",
                                       "train_loss", "train_acc",
                                       "valid_loss", "valid_acc"])
            df.to_excel(log_path)'''

        train_ls, valid_ls = [], []
        torch.manual_seed(seed)
        model = self.model().to(self.device)
        optimizer = self.optimizer(model.parameters(), lr=self.lr, betas=(0.9, 0.98))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        model.train()
        for epoch in range(train_num_epochs):
            with tqdm.tqdm(enumerate(train_dataloader, 0),
                           desc='Training Epoch_%s' % epoch,
                           total=len(train_dataloader)) as t:
                for i, data in t:
                    img1, img2, soft_bio1, soft_bio2, train_label = data
                    img1, img2, soft_bio1, soft_bio2, train_label = img1.to(self.device), img2.to(self.device), \
                                                                    soft_bio1.to(self.device), soft_bio2.to(self.device), \
                                                                    train_label.to(self.device)

                    output1, output2 = model(img1, img2)
                    output1 = get_fusion_result(fusion_method, output1, soft_bio1)
                    output2 = get_fusion_result(fusion_method, output2, soft_bio2)
                    loss_contrastive = self.loss_fcn(output1, output2, train_label)
                    optimizer.zero_grad()
                    loss_contrastive.backward()
                    optimizer.step()
                    train_dist = F.pairwise_distance(output1, output2)
                    train_predict_lb = (train_dist < 1).int()
                    train_accurancy = (train_predict_lb == train_label).float().mean().item()
                    train_ls.append(self.res(loss_contrastive.item(), train_accurancy))
            print('train_loss:%.6f' % train_ls[-1][0], 'train_acc:%.6f\n' % train_ls[-1][1])
            '''log = {
                "Fold number": seed,
                "Training epoch": epoch,
                "train_loss": f'{train_ls[-1][0]}:.4f',
                "train_acc":,
                "valid_loss":,
                "valid_acc":
            }'''

            count = 0
        with torch.no_grad():
            model.eval()
            with tqdm.tqdm(enumerate(valid_dataloader),
                           desc='iteration on valid set...',
                           total=len(valid_dataloader)) as t:
                for j, data1 in t:
                    img01, img02, soft_bio01, soft_bio02, valid_label = data1
                    img01, img02, soft_bio01, soft_bio02, valid_label = img01.to(self.device), img02.to(self.device), \
                                                                    soft_bio01.to(self.device), soft_bio02.to(self.device), \
                                                                    valid_label.to(self.device)

                    output01, output02 = model(img01, img02)
                    output01 = get_fusion_result(fusion_method, output01, soft_bio01)
                    output02 = get_fusion_result(fusion_method, output02, soft_bio02)
                    loss_contrastive = self.loss_fcn(output01, output02, valid_label)
                    valid_dist = F.pairwise_distance(output01, output02)
                    valid_predict_lb = (valid_dist < 1).int()
                    count += (valid_predict_lb == valid_label).float().sum()
                    valid_accurancy = count.item() / len(valid_dataloader.dataset)
                    valid_ls.append(self.res(loss_contrastive.item(), valid_accurancy))
            print('valid_loss:%.6f' % valid_ls[-1][0], 'valid_acc:%.6f' % valid_ls[-1][1])
        torch.save(model.state_dict(), f'model_state_dict/with_fusion/fusion_simple_append/model_{train_number_epochs}_{seed}.pkl')
        return train_ls, valid_ls

    def res(self, x, y):
        return x, y


@torch.no_grad()
def test_model(model, test_loader: DataLoader):
    model.eval()
    count = 0
    for img1, img2, _, _, lb in test_loader:
        img1, img2, lb = img1.to(t.device), img2.to(t.device), lb.to(t.device)
        out1, out2 = model(img1, img2)
        dist = F.pairwise_distance(out1, out2)
        p = (dist < 1).int()
        count += (p == lb).float().sum()
    return count.item() / len(test_loader.dataset)


if __name__ == '__main__':
    train_number_epochs = 2

    t = Trainer(CNN, learning_rate=2e-4, batch_size=32, use_cuda=True)
    #t.k_fold_iteration(train_number_epochs)

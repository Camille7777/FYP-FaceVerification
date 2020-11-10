import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import os

from model.cnn_model import CNN, ContrastiveLoss
from dataset.dataset import LfwDataset


class Trainer:
    def __init__(self, model, lr=1e-2, batch_size=64, use_cuda=True):
        self.model = model()
        self.device = torch.device('cuda:0') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        self.loss_fcn = ContrastiveLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.data_root = 'dataset/lfw_cropped/split_data/train'
        self.datasets = [LfwDataset(os.path.join(self.data_root, f'0{i}'))
                         for i in tqdm.tqdm(range(1, 11), desc='loading data...')]

    def get_10_folder_data(self, index):
        valid_dataset = self.datasets[index]
        train_dataset = sum([self.datasets[i] for i in range(10) if i != index])
        return valid_dataset, train_dataset

    def k_folder_iteration(self):
        for k in range(10):
            valid, train = self.get_10_folder_data(k)
            valid_loader = DataLoader(valid, batch_size=self.batch_size, shuffle=True)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)




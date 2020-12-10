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

train_number_epochs = 2


class Trainer:
    def __init__(self, model, learning_rate=1e-2, batch_size=32, use_cuda=True):
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



from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np

soft_biometrics_file_path='LFW_SoftBiometrics/LFW_ManualAnnotations.txt'
with open(soft_biometrics_file_path, 'r') as f:
    soft_biometrics = {line.split(' ', 1)[0]: line.split(' ', 1)[1].strip() for line in f.readlines()}


data_transform = transforms.Compose([transforms.Grayscale(1),
                                     transforms.Resize((64, 64), 2),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5], std=[0.5])
                                     ])


class LfwDataset(Dataset):
    def __init__(self, root_dir=None):
        if root_dir:
            self.data = ImageFolder(root_dir)
        else:
            self.data = None

    def __getitem__(self, index):
        """

        :param index: 表示数据的index
        :return: 返回两张图片,它们的soft_biometrics和label,label为1表示同一个人，为0表示非同一个人
        """
        img1 = self.data[2 * index][0]
        img2 = self.data[2 * index + 1][0]
        label = self.data[2 * index][1]
        img1_file = self.data.imgs[2 * index][0][self.data.imgs[2 * index][0].rfind('\\') + 1:]
        img2_file = self.data.imgs[2 * index + 1][0][self.data.imgs[2 * index + 1][0].rfind('\\') + 1:]
        img1_soft_biometrics = list(map(lambda x: int(x), soft_biometrics.get(img1_file).split(' ')))
        img2_soft_biometrics = list(map(lambda x: int(x), soft_biometrics.get(img2_file).split(' ')))
        return data_transform(img1), data_transform(img2), \
            torch.tensor(img1_soft_biometrics), torch.tensor(img2_soft_biometrics), \
            torch.tensor([label], dtype=torch.float32)

    def __add__(self, other):
        lfw_dataset = LfwDataset()
        lfw_dataset.data = self.data + other.data
        return lfw_dataset

    def __len__(self):
        return len(self.data) // 2


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def visualize(data_loader, batch_size=8):
    example_batch = next(iter(data_loader))
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated, nrow=batch_size))
    print(example_batch[2].numpy())

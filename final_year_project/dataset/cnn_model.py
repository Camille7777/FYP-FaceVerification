import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # convolution layer
            # Input Tensor Shape: [batch_size, 3, 64, 64]
            # Output Tensor Shape: [batch_size, 32, 64, 64]
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # pooling
            # Input Tensor Shape: [batch_size, 32, 64, 64]
            # Output Tensor Shape: [batch_size, 32, 32, 32]
            nn.MaxPool2d(2),

            # convolution layer
            # Input Tensor Shape: [batch_size, 32, 32, 32]
            # Output Tensor Shape: [batch_size, 64, 32, 32]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # pooling
            # Input Tensor Shape: [batch_size, 64, 32, 32]
            # Output Tensor Shape: [batch_size, 64, 16, 16]
            nn.MaxPool2d(2),

            # convolution layer
            # Input Tensor Shape: [batch_size, 64, 16, 16]
            # Output Tensor Shape: [batch_size, 64, 16, 16]
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # pooling
            # Input Tensor Shape: [batch_size, 64, 16, 16]
            # Output Tensor Shape: [batch_size, 64, 8, 8]
            nn.MaxPool2d(2)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(8*8*64, 1000)
        )

    def forward_single(self, img):
        output = self.cnn_layers(img)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

    def forward(self, img1, img2):
        output1 = self.forward_single(img1)
        output2 = self.forward_single(img2)
        return output1, output2

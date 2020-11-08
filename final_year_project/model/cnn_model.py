import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvUnit, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, img):
        return self.conv_layer(img)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            # Input Tensor Shape: [batch_size, 3, 64, 64]
            # Output Tensor Shape: [batch_size, 32, 32, 32]
            ConvUnit(1, 32, 5, 1, 2),
            # Input Tensor Shape: [batch_size, 32, 32, 32]
            # Output Tensor Shape: [batch_size, 64, 16, 16]
            ConvUnit(32, 64, 3, 1, 1),
            # Input Tensor Shape: [batch_size, 64, 16, 16]
            # Output Tensor Shape: [batch_size, 64, 8, 8]
            ConvUnit(64, 64, 3, 1, 1),
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

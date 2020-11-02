import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np


data_transform = transforms.Compose([transforms.Grayscale(1),
                                     transforms.Resize((64, 64), 2), #resize images(interpolation)
                                     transforms.RandomHorizontalFlip(p=0.5), #flip images
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])


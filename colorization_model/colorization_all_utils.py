# -*- coding: utf-8 -*-
# This code is here to simplify the import of the ColorizationNet class into app.py

#%% imports
import torch #requirement: pip install pytorch. Version depends on your build,
# if unsure, just install pytorch and it will run on the cpu instead of the gpu.
#import torch.nn as nn
from torch import nn

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np



#%% Define the colorization model
class ColorizationNet(nn.Module):
    """Convolutional Neural Network model for colorizing grayscale images.
    This class implements a convolutional neural network (CNN) to convert grayscale images
    to color images.
    
    Attributes:
        conv1 (nn.Conv2d): Convolutional layer 1.
        conv2 (nn.Conv2d): Convolutional layer 2.
        conv3 (nn.Conv2d): Convolutional layer 3.
        conv4 (nn.Conv2d): Convolutional layer 4 (Output layer).
    """

    def __init__(self):
        """Initializes the model with its convolutional layers."""
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4,
                               dilation=2)  # grayscale to 64 feature maps
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4,
                               dilation=2)  # 64 to 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4,
                               dilation=2)  # increase of feature maps
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1,
                               padding=4, dilation=2)  # 3 channels: RGB

    def forward(self, x):
        """Performs forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor representing a grayscale image.

        Returns:
            torch.Tensor: Tensor representing a color image.
        """
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))

        return x


def rgb_to_gray(img : Image.Image) -> Image.Image:
    """
    Function changing RGB images to grayscale
    
    Args :
        img : input RGB image
        
    Returns :
        Image.Image : grayscale image
    """
    return img.mean(dim=1, keepdim=True)

def imshow(img : Image.Image) -> None :
    """
    Convert from Tensor image and display

    Args : 
        img : image to display
    """

    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    if len(img.shape) == 2:  # grayscale image
        plt.imshow(npimg, cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# -*- coding: utf-8 -*-

"""
READ FIRST ! 

For this code to work, you first have to download and unzip the color images into 
the \images_folder\color\color_images folder.
"""

#%% imports

import os

import torch #requirement: pip install pytorch. Version depends on your build,
             # if unsure, just install pytorch and it will run on the cpu instead of the gpu.
#import torch.nn as nn
from torch import nn
#import torch.optim as optim
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import splitfolders #requirement: pip install split-folders==0.5.1
# from PIL import Image

from colorization_all_utils import ColorizationNet, rgb_to_gray

# %% Define the colorization model
"""
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4,
                               dilation=2)  # grayscale to 64 feature maps
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)  # 64 to 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4,
                               dilation=2)
                               # increase of feature maps to capture more complex patterns
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)  
        # 3 channels: RGB

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))

        return x
"""

"""
# Convert RGB image to grayscale
    def rgb_to_gray(img : Image.Image) -> Image.Image:
        
        Function changing RGB images to grayscale
        
        Args :
            img : input RGB image
            
        Returns :
            Image.Image : grayscale image
        
        return img.mean(dim=1, keepdim=True)
"""

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Load the dataset, split it into train and test.
# Fix the seed to have the same split for everybody.
    IMG_PATH=r".\images_folder\color"
    INP=IMG_PATH
    OUTPUT=r".\images_folder\split"

    splitfolders.ratio(INP, output=OUTPUT, seed=123, ratio=(.85, 0,0.15))

# %% Setup the transformations for the training and testing dataset.

    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])
    """
    TrivialAugment is a Data Augmentation method: an augmentation method is chosen
    in a range of trivial augmentations (change in lighting, image flip, etc.), 
    applied at a given magnitude.
    Read more here:
    https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html
    """

    test_transform = transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=OUTPUT+"\\train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root=OUTPUT+"\\test", transform=test_transform)
    # ImageFolder expects a folder with subfolders,
    # that are normally used to get the number of classification bins.
    # Here, there is no classification, hence there is only one subfolder.
    # %%
    train_loader = DataLoader(dataset=train_dataset, batch_size=1,
    num_workers=os.cpu_count(), shuffle=True)
    # num_workers=os.cpu_count() allows the DataLoader to use as much cpu cores as needed.

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, 
    num_workers=os.cpu_count(), shuffle=True)

    model = ColorizationNet().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Training loop
    EPOCHS = 15
    for epoch in range(EPOCHS):
        for i, (images, _) in enumerate(train_loader):
            grayscale_images = rgb_to_gray(images).to(device)
            images = images.to(device)

            # Forward pass
            outputs = model(grayscale_images)
            loss = criterion(outputs, images)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print statistics
            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print("Finished Training")

    # save model parameters

    # path of the file
    PARAMS_FILE_PATH = 'model1.pth'

    # save parameters in this file
    torch.save(model.state_dict(), PARAMS_FILE_PATH)

    print("Parameters saved")

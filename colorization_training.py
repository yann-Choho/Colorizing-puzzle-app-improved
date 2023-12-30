# -*- coding: utf-8 -*-

"""
READ FIRST ! 

For this code to work, you first have to download and unzip the color images into the \images_folder\color folder.

"""

#%% imports
import torch #requirement: pip install pytorch. Version depends on your build, if unsure, just install pytorch and it will run on the cpu instead of the gpu. 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import splitfolders #requirement: pip install split-folders==0.5.1
#%% Define the colorization model
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4,
                               dilation=2)  # grayscale to 64 feature maps
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)  # 64 to 64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4,
                               dilation=2)  # increase of feature maps to capture more complex patterns
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)  # 3 channels: RGB

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))

        return x


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Load the dataset, split it into train and test. Fix the seed to have the same split for everybody.
    img_path=r".\images_folder\color"
    # img_path = r"C:\Users\Thomas\Downloads\Colorisation-d-images-via-le-machine-learning-dev_scraping\images folder\color_planes"
    inp=img_path
    output=r".\images_folder\split"
    
    splitfolders.ratio(inp, output=output, seed=123, ratio=(.85, 0,0.15)) 
# %% Setup the transformations for the training and testing dataset.

    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()  
    ])
    # TrivialAugment is a Data Augmentation method: an augmentation method is chosed in a range of trivial augmentations (change in lighting, image flip, etc.), applied at a given magnitude. Read more here: https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html

    test_transform = transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=output+"\\train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root=output+"\\test", transform=test_transform)
    # ImageFolder expects a folder with subfolders, that are normally used to get the number of classification bins. Here, there is no classification, hence there is only one subfolder.
    # %%
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=os.cpu_count(), shuffle=True)
    # num_workers=os.cpu_count() allows the DataLoader to use as much cpu cores as needed.

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=os.cpu_count(), shuffle=True)

    model = ColorizationNet().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Convert RGB image to grayscale
    def rgb_to_gray(img):
        return img.mean(dim=1, keepdim=True)


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
    chemin_fichier_params = 'modele_1.pth' 

    # save parameters in this file
    torch.save(model.state_dict(), chemin_fichier_params)

    print("Parameters saved")



# -*- coding: utf-8 -*-
"""Copie de colorize.ipynb
CHECK LIGNE 263 pour la localisation de l'image test. Check ligne 64 pour le nombre d'Epochs.
"""
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
#%%

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
  print(torch.cuda.current_device())
  print(torch.cuda.device(0))
  print(torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))
else:
  print("No NVIDIA driver found. Using CPU")

#%% Load the CIFAR-10 dataset
img_path=r"C:\Users\rayan\Documents\GitHub\Projet infra\images folder\color"
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cette ligne indique la façon dont il faudra transformer les images.
data_transform = transforms.Compose([
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor() # normalise les valeurs des pixels.
    ])
# TrivialAugment est une méthode de Data Augmentation: on choisit une méthode d'augmentation parmi une sélection de méthodes triviales (baisse de luminosité, retournement de l'image, etc.), plus ou moins impactantes, et appliquées à un degré plus ou moins intense. num_magnitude bins, situé entre 1 et 31, augmente la probabilité d'avoir une transformation plus sérieuse et appliquée de façon plus intense.


full_dataset = datasets.ImageFolder(root=img_path, transform=data_transform, target_transform=None)
# attention: pour le chemin menant aux images, ImageFolder attend un dossier comportant des sous-dossiers, un pour chaque classe. Pour cet exercice, il n'est pas question de classifier les images, il n'y aura donc qu'un seul sous-dossier
#%%
train_subset, test_subset= random_split(full_dataset,[0.8,0.2])

train_loader = DataLoader(dataset=train_subset,batch_size=1,num_workers=os.cpu_count(),shuffle=True)
#num_workers=os.cpu_count() pour que le DataLoader puisse utiliser un maximum de processeurs pour charger les données.

test_loader = DataLoader(dataset=test_subset,batch_size=1,num_workers=os.cpu_count(),shuffle=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])

if __name__=="__main__":
    # Define the colorization model
    class ColorizationNet(nn.Module):
        def __init__(self):
            super(ColorizationNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
            self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)
    
        def forward(self, x):
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.relu(self.conv3(x))
            x = torch.sigmoid(self.conv4(x))
            return x
    
    model = ColorizationNet().to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert RGB image to grayscale
    def rgb_to_gray(img):
        return img.mean(dim=1, keepdim=True)
    
    # Training loop
    EPOCHS = 30
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
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    print("Finished Training")
#%%
    

    def imshow(img):
        # Convert from Tensor image and display
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        if len(img.shape) == 2:  # grayscale image
            plt.imshow(npimg, cmap='gray')
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    def visualize_all_three(original_images, grayscale_images, colorized_images, n=5):
        """
        Display grayscale, colorized, and original images side by side.
        n: number of images to display from the batch
        """
        fig = plt.figure(figsize=(3*n, 4))
        for i in range(n):
            # Display original image
            ax = plt.subplot(1, 3*n, 3*i + 1)
            imshow(original_images[i])
            ax.set_title("Original")
            ax.axis("off")
    
            # Display original grayscale image
            ax = plt.subplot(1, 3*n, 3*i + 2)
            imshow(grayscale_images[i])
            ax.set_title("Grayscale")
            ax.axis("off")
    
            # Display colorized image
            ax = plt.subplot(1, 3*n, 3*i + 3)
            imshow(colorized_images[i])
            ax.set_title("Colorized")
            ax.axis("off")
    
        plt.tight_layout()
        plt.show()
    
    
    def torch_rgb_to_hsv(rgb):
        """
        Convert an RGB image tensor to HSV.
    
        Parameters:
        - rgb: tensor of shape (batch_size, 3, height, width) in RGB format in the range [0, 1].
    
        Returns:
        - hsv: tensor of same shape in HSV format in the range [0, 1].
        """
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
        max_val, _ = torch.max(rgb, dim=1)
        min_val, _ = torch.min(rgb, dim=1)
        diff = max_val - min_val
    
        # Compute H
        h = torch.zeros_like(r)
        mask = (max_val == r) & (g >= b)
        h[mask] = (g[mask] - b[mask]) / diff[mask]
        mask = (max_val == r) & (g < b)
        h[mask] = (g[mask] - b[mask]) / diff[mask] + 6.0
        mask = max_val == g
        h[mask] = (b[mask] - r[mask]) / diff[mask] + 2.0
        mask = max_val == b
        h[mask] = (r[mask] - g[mask]) / diff[mask] + 4.0
        h = h / 6.0
        h[diff == 0.0] = 0.0
    
        # Compute S
        s = torch.zeros_like(r)
        s[diff != 0.0] = diff[diff != 0.0] / max_val[diff != 0.0]
    
        # V is just max_val
        v = max_val
    
        return torch.stack([h, s, v], dim=1)
    
    
    def torch_hsv_to_rgb(hsv):
        """
        Convert an HSV image tensor to RGB.
    
        Parameters:
        - hsv: tensor of shape (batch_size, 3, height, width) in HSV format in the range [0, 1].
    
        Returns:
        - rgb: tensor of same shape in RGB format in the range [0, 1].
        """
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        i = (h * 6.0).floor()
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
    
        i_mod = i % 6
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
    
        r[i_mod == 0.0] = v[i_mod == 0.0]
        g[i_mod == 0.0] = t[i_mod == 0.0]
        b[i_mod == 0.0] = p[i_mod == 0.0]
    
        r[i_mod == 1.0] = q[i_mod == 1.0]
        g[i_mod == 1.0] = v[i_mod == 1.0]
        b[i_mod == 1.0] = p[i_mod == 1.0]
    
        r[i_mod == 2.0] = p[i_mod == 2.0]
        g[i_mod == 2.0] = v[i_mod == 2.0]
        b[i_mod == 2.0] = t[i_mod == 2.0]
    
        r[i_mod == 3.0] = p[i_mod == 3.0]
        g[i_mod == 3.0] = q[i_mod == 3.0]
        b[i_mod == 3.0] = v[i_mod == 3.0]
    
        r[i_mod == 4.0] = t[i_mod == 4.0]
        g[i_mod == 4.0] = p[i_mod == 4.0]
        b[i_mod == 4.0] = v[i_mod == 4.0]
    
        r[i_mod == 5.0] = v[i_mod == 5.0]
        g[i_mod == 5.0] = p[i_mod == 5.0]
        b[i_mod == 5.0] = q[i_mod == 5.0]
    
        return torch.stack([r, g, b], dim=1)
    
    def exaggerate_colors(images, saturation_factor=1.5, value_factor=1.2):
        """
        Exaggerate the colors of RGB images.
    
        Parameters:
        - images: tensor of shape (batch_size, 3, height, width) in RGB format.
        - saturation_factor: factor by which to increase the saturation. Default is 1.5.
        - value_factor: factor by which to increase the value/brightness. Default is 1.2.
    
        Returns:
        - color_exaggerated_images: tensor of same shape as input, with exaggerated colors.
        """
        # Convert images to the range [0, 1]
        images = (images + 1) / 2.0
    
        # Convert RGB images to HSV
        images_hsv = torch_rgb_to_hsv(images)
    
        # Increase the saturation and value components
        images_hsv[:, 1, :, :] = torch.clamp(images_hsv[:, 1, :, :] * saturation_factor, 0, 1)
        images_hsv[:, 2, :, :] = torch.clamp(images_hsv[:, 2, :, :] * value_factor, 0, 1)
    
        # Convert the modified HSV images back to RGB
        color_exaggerated_images = torch_hsv_to_rgb(images_hsv)
    
        # Convert images back to the range [-1, 1]
        color_exaggerated_images = color_exaggerated_images * 2.0 - 1.0
    
        return color_exaggerated_images
    
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            grayscale_images = rgb_to_gray(images).to(device)
            colorized_images = model(grayscale_images)
    
            # Convert the tensors back to CPU for visualization
            grayscale_images_cpu = grayscale_images.cpu().squeeze(1)  # remove the color channel
            colorized_images_cpu = colorized_images.cpu()
            original_images_cpu = images.cpu()
    
            #colorized_images_cpu=scale_predicted_colors(colorized_images_cpu)
            colorized_images_cpu=exaggerate_colors(colorized_images_cpu)
    
            # Visualize the grayscale, colorized, and original images
            # visualize_all_three(original_images_cpu, grayscale_images_cpu, colorized_images_cpu)
    
            if i == 10:  # only do this for up to certain batch for demonstration purposes
                break
    
    
#%%
    # Open the image. (Keep your image in the current directory. In my case, the image was horse.jpg)
    img = Image.open(r"C:\Users\rayan\Documents\GitHub\Projet infra\images folder\montagne-ete-scaled.jpg")
    
    # Convert the image to grayscale
    gray_img = img.convert("L")
        
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # If you need to normalize, uncomment the following line
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming you want to normalize to [-1, 1] range
    ])
    
    # Apply the transformations
    img_tensor = transform(gray_img).unsqueeze(0)  # Add a batch dimension
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Move the image tensor to the device where your model is (likely 'cuda' if using GPU)
    img_tensor = img_tensor.to(device)
    
    # Get the model's output
    with torch.no_grad():
        colorized_tensor = model(img_tensor)
    
    # Convert the tensor back to an image
    colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())
    
    
    # Plotting the original, grayscale, and colorized images side-by-side
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # Create a figure with 1 row and 3 columns
    
    # Display original color image
    ax[0].imshow(img)
    ax[0].set_title("Original Color Image")
    ax[0].axis('off')  # Hide axes
    
    # Display grayscale image
    ax[1].imshow(gray_img, cmap='gray')  # Since it's grayscale, use cmap='gray'
    ax[1].set_title("Grayscale Image")
    ax[1].axis('off')  # Hide axes
    
    # Display colorized image
    ax[2].imshow(colorized_img)
    ax[2].set_title("Colorized Image")
    ax[2].axis('off')  # Hide axes
    
    plt.tight_layout()  # Adjust spacing
    plt.show()
    
   

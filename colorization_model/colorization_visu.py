# -*- coding: utf-8 -*-

"""
This code is here to show how the model colorizes an image.
Run colorization_training first, expecially the splitfolders part, if you want to see it run on the leftover images from the training dataset.

"""
#%%
import torch
# import torch.nn as nn
# import torch.optim as optim
from torchvision import transforms
# from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from colorization_training import ColorizationNet
#%%

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define the model
model = ColorizationNet().to(device)


# load optimal parameters for the model
print(os.getcwd())
#%%
# chemin d' acces a changer selon besoins
path_params = os.path.dirname(os.getcwd())
model.load_state_dict(torch.load(path_params))

print("Model set up with optimal parameters provided")
# %%
def imshow(img):
    # Convert from Tensor image and display
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    if len(img.shape) == 2:  # grayscale image
        plt.imshow(npimg, cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
#%% showing the colorization of a random image from our database.

output=r"..\images_folder\split"
try:
    images = os.listdir(output+"\\test\color_images") #This folder is created if colorization_training is ran.
except FileNotFoundError:
    images=[]
    
if len(images)>0:
    i=np.random.randint(0,len(images))
    
    img_str = os.path.join(output+"\\test\color_images",images[i])
    # Convert the image to grayscale
    img = Image.open(img_str)
    
    gray_img = img.convert("L")
    
    # Turn it into a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(gray_img).unsqueeze(0)  # Add a batch dimension
    
    # Move the image tensor to the device where your model is
    img_tensor = img_tensor.to(device)
    
    # Get the model's output
    with torch.no_grad():
        colorized_tensor = model(img_tensor)
    
    # Convert the tensor back to an image
    colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())
      
    
    # Plotting the original, grayscale, and colorized images side-by-side
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
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

#%% showing the colorization of a random image from the Internet.
img = Image.open(r".\images folder\Fleurs-des-champs.jpeg")
    
gray_img = img.convert("L")

# Turn it into a tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

img_tensor = transform(gray_img).unsqueeze(0)  
img_tensor = img_tensor.to(device)

# Get the model's output
with torch.no_grad():
    colorized_tensor = model(img_tensor)


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


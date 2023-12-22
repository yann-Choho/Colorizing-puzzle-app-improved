# -*- coding: utf-8 -*-

# imports

#%%
import torch
# import torch.nn as nn
# import torch.optim as optim
from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# import os

from colorization_training import ColorizationNet
#%%



if __name__ == "__main__" :

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # define the model
    model = ColorizationNet().to(device)


    # load optimal parameters for the model

    # chemin d' acces a changer selon besoins
    path_params = 'modele_1.pth'
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


    # %%
    # Open the image. (Keep your image in the current directory. In my case, the image was horse.jpg)
    # img = Image.open(r"C:\Users\rayan\Documents\GitHub\Projet infra\images folder\screen.jpg")
    img = Image.open(r"C:\Users\Thomas\Downloads\Colorisation-d-images-via-le-machine-learning-dev_scraping\images folder\montagne_test.jpg")

    # Convert the image to grayscale
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

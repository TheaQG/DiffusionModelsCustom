'''
    First attempt on diffusion model for HR data generation
'''


# Import libraries
import random 
import imageio
import numpy as np
from argparse import ArgumentParser

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

#import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Compose, Lambda
from torchvision.datasets.mnist import MNIST, FashionMNIST

# Set reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths to store data

STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
STORE_PATH_FASHION = f"ddpm_model_fashion.pt"




# Set some hyperparameters

no_train = False 
fashion = False
batch_size = 128
n_epochs = 20
lr = 0.01 
store_path = "ddpm_fashion.pt" if fashion else "ddpm_mnist.pt"



# Make a function that can display the images 

def show_images(images, title=''):
    '''
        Show selected images in a square with subgrids
    '''

    # Convert images to CPU np arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Define number rows and columns
    fig = plt.figure(figsize=(8,8))
    rows = int(len(images) ** 0.5)
    cols = round(len(images) // rows)

    # Show images as subplots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx+1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap='gray')
                idx += 1

    fig.suptitle(title, fontsize=30)
    plt.show()
    return 

# Test the function by displaying the first batch of MNIST images

def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], 'Images in first batch')
        break

# Load data, normalize btw [-1,1] and show first batch

transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
)

ds_fn = FashionMNIST if fashion else MNIST
dataset = ds_fn("./datasets", train=True, download=True, transform=transform)   
loader = DataLoader(dataset, batch_size, shuffle=True)

show_first_batch(loader)



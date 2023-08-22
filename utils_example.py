import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    '''
        Plot a grid of images
    '''
    plt.figure(figsize=(32,32)) # Set figure size
    plt.imshow(torch.cat([ # Concatenate images
        torch.cat([i for i in images.cpu()], dim=-1), # Concatenate images in row
    ], dim=-2).permute(1,2,0).cpu()) # Concatenate images in column
    plt.show() # Show image


def save_images(images, path, **kwargs):
    '''
        Save a grid of images
    '''
    grid = torchvision.utils.make_grid(images, **kwargs) # Make grid
    ndarr = grid.permute(1,2,0).cpu().numpy() # Convert to numpy array
    im = Image.fromarray(ndarr) # Convert to PIL image

    im.save(path) # Save image


def get_data(args):
    '''
        Get data from dataset path and return dataloader
    '''
    # Define transforms for data augmentation and normalization 
    transforms = torchvision.transforms.Compose([ 
        torchvision.transforms.Resize(80), # Resize to 80x80
        torchvision.transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)), # Random crop
        torchvision.transforms.ToTensor(), # Convert to tensor
        torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # Normalize to [-1,1]
    ])

    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms) # Load dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) # Create dataloader

    return dataloader

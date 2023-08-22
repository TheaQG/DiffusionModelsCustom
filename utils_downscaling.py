import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def show_images(images, title=""):
    
    # Convert to cpu numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Set number of rows, columns
    fig = plt.figure(figzie=(8,8))
    rows = int(len(images) ** (1/2))
    cols = len(images) // rows

    # Plot images in sub plots 
    idx = 0 

    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, idx+1)

            if idx < len(images):
                plt.imshow(images[idx][0])
                idx += 1

    fig.suptitle(title, fontsize=30)

    plt.show()


def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], 'Images in first batch')
        break

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1,2,0).cpu().numpy()
    im = Image.fromarray(ndarr)

    im.save(path)


def get_data(args):
    dataLoader = torch.zeros(5)
    return dataLoader
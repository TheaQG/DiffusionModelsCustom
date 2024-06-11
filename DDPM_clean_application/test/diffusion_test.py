'''
    Short script to test the diffusion module.

    Checks the forward diffusion process

'''

import torch
import logging
import zarr

import torch.nn as nn
import matplotlib.pyplot as plt

from ..src.diffusion_modules import DiffusionUtils
from ..src.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
# Import freeze_support from multiprocessing
from multiprocessing import freeze_support

if __name__ == '__main__':
    # Use multiprocessing freeze_support() to avoid RuntimeError:
    freeze_support()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set number of timesteps
    n_timesteps = 1000

    # Set minimum and maximum beta values
    beta_min = 1e-4
    beta_max = 0.02

    # Set scheduler
    scheduler = 'cosine'

    # Set image size
    img_size = 128
    image_size = (img_size, img_size)

    # Create a simple unet model (just because we need a model to test the diffusion process)
    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv4 = nn.Conv2d(64, 3, 3, 1, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.conv4(x)
            return x
        
    # Initialize the model
    model = UNet().to(device)

    # Initialize the diffusion module
    diffusion = DiffusionUtils(n_timesteps=n_timesteps,
                            beta_min=beta_min,
                            beta_max=beta_max,
                            device=device,
                            scheduler=scheduler,
                            img_size=img_size)

    # Set number of samples
    n_samples = 10

    var = 'temp' #'prcp' # 

    # Get data (no conditions)
    data_dir_danra_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789_full/zarr_files/' + var + '_589x789_test.zarr'
    danra_w_cutouts_zarr_group = zarr.open(data_dir_danra_w_cutouts_zarr, mode='r')

    dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_danra_w_cutouts_zarr,
                                        image_size, 
                                        n_samples, 
                                        cache_size=n_samples, 
                                        variable = var,
                                        shuffle=True, 
                                        cutouts = True, 
                                        cutout_domains = [170, 170+180, 340, 340+180],
                                        n_samples_w_cutouts = n_samples, 
                                        scale=False,
                                        conditional_seasons=False,
                                        conditional_images=False,
                                        )


    # Get one sample
    x = dataset[0]['img']

    # Stack n_samples of the same image
    x = x.repeat(n_samples, 1, 1, 1)
    print(x.shape)
    # Test the forward diffusion process (i.e. the noiseImage method)
    x = x.to(device)
    
    # Get 10 evenly spaced timesteps from 1 to n_timesteps and noise image at each timestep
    t = torch.linspace(99, n_timesteps-1, n_samples).long().to(device)

    # Get noised images
    x_hat, noise = diffusion.noiseImage(x, t)
    
    # Plot the noised images
    fig, ax = plt.subplots(2, 5, figsize=(20, 5))
    fig.suptitle('Noised images')
    for i, ax in enumerate(ax.flatten()):
        ax.imshow(x_hat[i].detach().cpu().permute(1, 2, 0))
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f'Timestep {t[i]+1}/{n_timesteps}')


    # Test the reverse diffusion process (sample() method)

    x_tilde = diffusion.sample(n_samples, model, channels_hr=1, y=None, cond_img=None, lsm_cond=None, topo_cond=None, cfg_scale=0.0)

    plt.show()

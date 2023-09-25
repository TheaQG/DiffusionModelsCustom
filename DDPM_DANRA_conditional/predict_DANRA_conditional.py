import os, tqdm, random, torch
import numpy as np
import torch.nn as nn
#from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Union, Iterable, Tuple

from diffusion_DANRA_conditional import DiffusionUtils
from modules_DANRA_conditional import *
from data_DANRA_conditional import DANRA_Dataset, preprocess_lsm_topography
from training_DANRA_conditional import TrainingPipeline

PATH_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures'

# Define DANRA data information 
# Set variable for use
var = 'temp'#'prcp'# 
# Set size of DANRA images
n_danra_size = 128# 256#
# Set DANRA size string for use in path
danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)
# Set paths to data
data_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str

n_files = 0
for root, _, files in os.walk(data_dir_danra):
    for name in files:
        if name.endswith('.npz'):
            n_files += 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



n_timesteps = 800
beta_min = 1e-4
beta_max = 0.02
beta_scheduler = 'linear'

# Define data hyperparameters
input_channels = 1
output_channels = 1
n_samples = n_files
cache_size = n_files
image_size = (64,64)#(32,32)#(n_danra_size, n_danra_size)#
n_seasons = 4

# Define model hyperparameters
epochs = 60
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
first_fmap_channels = 64#n_danra_size #
last_fmap_channels = 512 #2048
time_embedding = 256
learning_rate = 3e-4 #1e-2
min_lr = 1e-6
weight_decay = 0.0


# Preprocess lsm and topography (normalize and reshape, resizes to image_size)
PATH_LSM = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_DK/lsm_dk.npz'
PATH_TOPO = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_DK/topo_dk.npz'
lsm_tensor, topo_tensor = preprocess_lsm_topography(PATH_LSM, PATH_TOPO, image_size)
lsm_weight = 1
lsm_tensor = lsm_weight * lsm_tensor
topo_weight = 1
topo_tensor = topo_weight * topo_tensor


encoder = Encoder(input_channels, time_embedding, lsm_tensor=lsm_tensor, topo_tensor=topo_tensor, block_layers=[2, 2, 2, 2], num_classes=4)
decoder = Decoder(last_fmap_channels, output_channels, time_embedding, first_fmap_channels)
# Define the model from modules_DANRA_downscaling.py
model = DiffusionNet(encoder, decoder)


diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, device, beta_scheduler)



checkpoint_dir = '../../ModelCheckpoints/DDPM_DANRA/'
checkpoint_name = 'DDPM_conditional__lsm_topo__nonScaled.pth.tar'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

best_model_path = os.path.join('../../ModelCheckpoints/DDPM_DANRA', 'DDPM_conditional__lsm_topo__nonScaled.pth.tar')
best_model_state = torch.load(best_model_path)['network_params']


# Define the loss function
lossfunc = nn.MSELoss()
# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Define the training pipeline from training_DANRA_downscaling.py
pipeline = TrainingPipeline(model, lossfunc, optimizer, diffusion_utils, device, weight_init=True)

# Load best model state into model0
pipeline.model.load_state_dict(best_model_state)

print('Generating samples...')

# Set number of samples to generate
n = 8

# Generate samples
x = torch.randn(n, input_channels, *image_size).to(device)

# Generate random season labels of batchsize n
y = torch.randint(0, n_seasons, (n,)).to(device) # 4 seasons, 0-3

# Sample generated images from model
generated_images = diffusion_utils.sample(x, pipeline.model, y)
generated_images = generated_images.detach().cpu()

# Plot samples
fig, axs = plt.subplots(1, n, figsize=(18,3))

for i in range(n):
    img = generated_images[i].squeeze()
    image = axs[i].imshow(img, cmap='viridis')
    axs[i].set_title(f'Season: {y[i].item()}')
    axs[i].axis('off')
    fig.colorbar(image, ax=axs[i], fraction=0.046, pad=0.04)
fig.tight_layout()

fig.savefig(PATH_SAVE + f'/Samples/Samples_64x64__lsm_topo__nonScaled/predicted__{var}_generated_samples_64x64_2.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()

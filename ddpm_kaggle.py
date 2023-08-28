import os, tqdm, random, torch
import numpy as np
import torch.nn as nn
from PIL import Image
from multiprocessing import Manager as SharedMemoryManager
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Union, Iterable, Tuple

# Set DANRA variable for use
var = 'temp'#'prcp'#
# Set size of DANRA images
n_danra_size = 256#128#
# Set DANRA size string for use in path
danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)

# Set paths to data
data_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str

SAVE_FIGS = False
PATH_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_channels = 1
first_fmap_channels = 64
last_fmap_channels = 512 #2048
output_channels = 3
time_embedding = 256
learning_rate = 1e-4 #1e-2
min_lr = 1e-6
weight_decay = 0.0
n_timesteps = 550
beta_min = 1e-4
beta_max = 2e-2
beta_scheduler = 'linear'
batch_size = 10
n_samples = 365
cache_size = 365
image_size = (n_danra_size, n_danra_size)
seed = random.randint(0, 2**32 - 1)

from data import DANRA_Dataset

dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size, seed=seed)

sample_img = dataset[0]

print(f'shape: {sample_img.shape}')
print(f'min pixel value: {sample_img.min()}')
print(f'mean pixel value: {sample_img.mean()}')
print(f'max pixel value: {sample_img.max()}')

print(seed)

plt.imshow(sample_img.squeeze())


class DiffusionUtils:
    def __init__(self, n_timesteps:int, beta_min:float, beta_max:float, device:str='cpu', scheduler:str='linear'):
        assert scheduler in ['linear', 'cosine'], 'scheduler must be linear or cosine'
        self.n_timesteps = n_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        self.scheduler = scheduler

        self.betas = self.betaSamples()
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def betaSamples(self):
        if self.scheduler == 'linear':
            return torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.n_timesteps).to(self.device)
        
        elif self.scheduler == 'cosine':
            betas = []
            for i in reversed(range(self.n_timesteps)):
                T = self.n_timesteps - 1
                beta = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (1 + np.cos((i / T) * np.pi))
                betas.append(beta)

            return torch.Tensor(betas).to(self.device)
        
    def sampleTimesteps(self, size:int):
        return torch.randint(low=1, high=self.n_timesteps, size=(size,)).to(self.device)
    
    def noiseImage(self, x:torch.Tensor, t:torch.LongTensor):
        assert len(x.shape) == 4, 'x must be a 4D tensor'

        alpha_hat_sqrts = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        one_minus_alpha_hat_sqrt = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x).to(self.device)
        
        return (alpha_hat_sqrts * x) + (one_minus_alpha_hat_sqrt * noise), noise
    
    def sample(self, x:torch.Tensor, model:nn.Module):
        #x shape: (batch_size, channels, height, width)
        assert len(x.shape) == 4, 'x must be a 4D tensor'
        
        model.eval()

        with torch.no_grad():
            iterations = range(1, self.n_timesteps)

            for i in tqdm.tqdm(reversed(iterations)):
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)

                alpha = self.alphas[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                one_minus_alpha = 1 - alpha
                one_minus_alpha_hat = 1 - alpha_hat

                predicted_noise = model(x, t)

                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                else:
                    noise = torch.zeros_like(x).to(self.device)


                x = (1 / torch.sqrt(alpha)) * (x - ((one_minus_alpha) / torch.sqrt(one_minus_alpha_hat)) * predicted_noise)
                x = x + (torch.sqrt(beta) * noise)

            return x
        


T = n_timesteps
n_steps = 50
alpha_values = {}

print('\n')
for scheduler in ['linear', 'cosine']:
    print(f'Running {scheduler} beta scheduler...')
    
    diffusion = DiffusionUtils(T, beta_min, beta_max, scheduler=scheduler)
    alpha_values[scheduler] = diffusion.alphas

    fig, axs = plt.subplots(2, (T//(n_steps*2))+1, figsize=(15,8))
    
    axs.flatten()[0].imshow(sample_img.squeeze())
    axs.flatten()[0].set_title('Original Image')

    for idx, t in enumerate(range(n_steps-1, T, n_steps)):
        t = torch.Tensor([t]).long()
        x, _ = diffusion.noiseImage(sample_img.unsqueeze(0), t)
        axs.flatten()[idx+1].imshow(x.squeeze())
        axs.flatten()[idx+1].set_title(f'Image at t={t.item()}')

    fig.set_tight_layout(True)
    if SAVE_FIGS:
        fig.savefig(PATH_SAVE + f'/ddpm_kaggle_ex__{scheduler}_diffusion.png')

    print('\n')



fig, axs = plt.subplots(1, 2, figsize=(20, 4))

axs[0].plot(alpha_values['linear'])
axs[0].set_xlabel('timestep (t)')
axs[0].set_ylabel('alpha (1-beta)')
axs[0].set_title('alpha values of linear scheduling')

axs[1].plot(alpha_values['cosine'])
axs[1].set_xlabel('timestep (t)')
axs[1].set_ylabel('alpha (1-beta)')
axs[1].set_title('alpha values of cosine scheduling')

plt.show()





class SiunsoidalEmbedding(nn.Module):
    def __init__(self, dim_size, n:int = 10000):
        assert dim_size % 2 == 0, 'dim_size must be even'

        super(SiunsoidalEmbedding, self).__init__()

        self.dim_size = dim_size
        self.n = n

    def forward(self, x:torch.Tensor):
        N = len(x)
        output = torch.zeros(size = (N, self.dim_size)).to(x.device)

        for idx in range(0,N):
            for i in range(0, self.dim_size//2):
                emb = x[idx] / (self.n ** (2*i / self.dim_size))
                output[idx, 2*i] = torch.sin(emb)
                output[idx, 2*i+1] = torch.cos(emb)

        return output
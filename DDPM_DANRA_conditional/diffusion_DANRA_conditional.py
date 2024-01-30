import tqdm, torch
import numpy as np
import torch.nn as nn
from modules_DANRA_conditional import *

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    print('Testing diffusion class')



class DiffusionUtils:
    '''
        Class for diffusion process of images by DDPM as described in the paper.
        The class is based on the implementation of the paper: 
        Diffusion is gaussian.
    '''
    def __init__(self,
                 n_timesteps:int,
                 beta_min:float,
                 beta_max:float,
                 device:str='cpu',
                 scheduler:str='linear'
                 ):
        '''
            Initialize the class.
            Input:
                - n_timesteps: number of timesteps
                - beta_min: minimum beta value
                - beta_max: maximum beta value
                - device: device to use (cpu or cuda)
                - scheduler: beta scheduler to use (linear or cosine)

        '''
        # Check if scheduler is valid
        assert scheduler in ['linear', 'cosine'], 'scheduler must be linear or cosine'

        # Set class variables
        self.n_timesteps = n_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        self.scheduler = scheduler

        # Set betas as sampled from scheduler
        self.betas = self.betaSamples().to(self.device)
        # Set alphas as defined in paper
        self.alphas = 1 - self.betas
        # Set alpha_hat as cumulative product of alphas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def betaSamples(self):
        '''
            Function for sampling betas from scheduler.
            Samples uniformly from scheduler.
            Cosine or linear. The value is sampled given a timestep.
        '''

        # Linear scheduler - samples linearly uniform
        if self.scheduler == 'linear':
            return torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.n_timesteps).to(self.device)
        
        # Cosine scheduler
        elif self.scheduler == 'cosine':
            
            betas = []
            # Loop over timesteps, reverse order
            for i in reversed(range(self.n_timesteps)):
                # Set current timestep
                T = self.n_timesteps - 1
                # Sample beta from cosine scheduler as defined in paper
                beta = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (1 + np.cos((i / T) * np.pi))
                # Append to list
                betas.append(beta)
            # Return list as tensor (forward order)
            return torch.Tensor(betas).to(self.device)
        
    def sampleTimesteps(self, size:int):
        '''
            Randomly sample timesteps from 1 to n_timesteps.
        '''
        return torch.randint(low=1, high=self.n_timesteps, size=(size,)).to(self.device)
    
    def noiseImage(self, x:torch.Tensor, t:torch.LongTensor):
        '''
            Function for sampling noise from gaussian distribution.
            Input:
                - x: image tensor
                - t: timestep tensor
        '''
        # Check if x is 4D tensor
        assert len(x.shape) == 4, 'x must be a 4D tensor'

        # Set alpha_hat_sqrt and one_minus_alpha_hat_sqrt
        alpha_hat_sqrts = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        one_minus_alpha_hat_sqrt = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        # Sample noise from gaussian distribution with same shape as x
        noise = torch.randn_like(x).to(self.device)
        
        # Return noise as defined in paper, and noise
        return (alpha_hat_sqrts * x) + (one_minus_alpha_hat_sqrt * noise), noise
    
    def sample(self,
               x:torch.Tensor,
               model:nn.Module,
               y:torch.Tensor=None,
               cond_img:torch.Tensor=None,
               lsm_cond:torch.Tensor=None,
               topo_cond:torch.Tensor=None):
        '''
            Function for sampling from diffusion process.
            Input:
                - x: image tensor
                - model: model to use for sampling
        '''
        # Check if x is 4D tensor, (batch_size, channels, height, width)
        assert len(x.shape) == 4, 'x must be a 4D tensor'

        # Set model to eval mode
        model.eval()

        # Set model to no grad mode for sampling process (no gradients are calculated)
        with torch.no_grad():
            # Set iterations as range from 1 to n_timesteps
            iterations = range(1, self.n_timesteps)

            # Loop over iterations in reverse order (from n_timesteps to 1)
            for i in tqdm.tqdm(reversed(iterations)):
                # Set timestep as current iteration, shape (batch_size,)
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)

                # Set alpha, beta, alpha_hat and one_minus_alpha_hat as defined in paper
                alpha = self.alphas[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                one_minus_alpha = 1 - alpha
                one_minus_alpha_hat = 1 - alpha_hat

                # Set predicted noise as output from model
                #
                # if y is None:
                #     predicted_noise = model(x, t)
                # else:
                predicted_noise = model(x, t, y, cond_img, lsm_cond, topo_cond)

                # Sample noise from gaussian distribution or set to zero if timestep is 1
                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                else:
                    noise = torch.zeros_like(x).to(self.device)

                # Compute x based on the 
                x = (1 / torch.sqrt(alpha)) * (x - ((one_minus_alpha) / torch.sqrt(one_minus_alpha_hat)) * predicted_noise)
                # Add noise to x
                x = x + (torch.sqrt(beta) * noise)

            return x
        



import tqdm, torch
import numpy as np
import torch.nn as nn
import logging

class DiffusionUtils:
    '''
        Class for diffusion process of images by DDPM as described in the paper.
        The class is based on the implementation of the paper: 
        Diffusion is gaussian.
    '''
    def __init__(self,
                 n_timesteps:int=1000,
                 beta_min:float=1e-4,
                 beta_max:float=0.02,
                 device:str='cpu',
                 scheduler:str='linear',
                 img_size:int=64,
                 data_scaled:bool=False
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
        self.img_size = img_size
        self.data_scaled = data_scaled

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
            
            t = torch.linspace(0, self.n_timesteps, self.n_timesteps + 1)
            ft = torch.cos(((t / self.n_timesteps + 0.008) / 1.008) * np.pi / 2)**2 # 0.008 and 1.008 are used to avoid division by zero
            alphat = ft / ft[0] # Normalize to start at 1
            betat = 1 - alphat[1:] / alphat[:-1] # Compute beta from alpha 
            return torch.clip(betat, 0.0001, 0.9999).to(self.device)
        
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
        assert len(x.shape) == 4, f'x must be a 4D tensor but is {len(x.shape)}D tensor.'

        # Set alpha_hat_sqrt and one_minus_alpha_hat_sqrt
        alpha_hat_sqrts = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        one_minus_alpha_hat_sqrt = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        # Sample noise from gaussian distribution with same shape as x
        noise = torch.randn_like(x).to(self.device)

        # If data is scaled, sample smaller variance noise
        if self.data_scaled:
            noise *= 0.005
        
        # Return noise as defined in paper, and noise
        return (alpha_hat_sqrts * x) + (one_minus_alpha_hat_sqrt * noise), noise
    
    def sample(self,
               n:int,
               model:nn.Module,
               channels_hr:int,
               y:torch.Tensor=None,
               cond_img:torch.Tensor=None,
               lsm_cond:torch.Tensor=None,
               topo_cond:torch.Tensor=None,
               cfg_scale:float=0.0):
        '''
            Function for sampling from diffusion process.
            Input:
                - n: number of images to sample
                - model: model to use for sampling
                - channels_hr: number of channels in high resolution image
                - y: category tensor (season)
                - cond_img: LR conditional image tensor
                - lsm_cond: LSM conditional tensor
                - topo_cond: Topological conditional tensor
                - cfg_scale: Scale for classifier free guidance
        '''
        #logging.info()
        # Set model to eval mode
        model.eval()

        # Set model to no grad mode for sampling process (no gradients are calculated)
        with torch.no_grad():
            # Set x as random noise, with shape (n, channels_hr, img_size, img_size)
            x = torch.randn((n, channels_hr, self.img_size, self.img_size)).to(self.device)

            if self.data_scaled:
                x *= 0.005

            # Send conditions to device if not None
            if cond_img is not None:
                cond_img = cond_img.to(self.device)
            if lsm_cond is not None:
                lsm_cond = lsm_cond.to(self.device)
            if topo_cond is not None:
                topo_cond = topo_cond.to(self.device)
            if y is not None:
                y = y.to(self.device)

            # Set iterations as range from 1 to n_timesteps
            iterations = range(1, self.n_timesteps)

            # Loop over iterations in reverse order (from n_timesteps to 1)
            for i in tqdm.tqdm(reversed(iterations), position=0, total=self.n_timesteps - 1):
                # Set timestep as current iteration, shape (batch_size,)
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)

                # Predict noise with given model, conditioned on various conditions
                predicted_noise = model(x, t, y, cond_img, lsm_cond, topo_cond)

                # Comput unconditional noise if cfg_scale is not zero
                if cfg_scale > 0:
                    # Predict noise unconditionally
                    uncond_predicted_noise = model(x, t, y=None, cond_img=None, lsm_cond=None, topo_cond=None)
                    # Linearly interpolate between conditional and unconditional noise
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                # Set alpha, beta, alpha_hat and one_minus_alpha_hat as defined in paper
                alpha = self.alphas[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                one_minus_alpha = 1 - alpha
                one_minus_alpha_hat = 1 - alpha_hat


                # Sample noise from gaussian distribution or set to zero if timestep is 1
                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                    if self.data_scaled:
                        noise *= 0.005
                else:
                    noise = torch.zeros_like(x).to(self.device)

                # Compute x based on the 
                x = (1 / torch.sqrt(alpha)) * (x - ((one_minus_alpha) / torch.sqrt(one_minus_alpha_hat)) * predicted_noise)
                # Add noise to x
                x = x + (torch.sqrt(beta) * noise)
        model.train()

        # Convert back to original scale?

        return x
        


'''
    Based on https://www.youtube.com/watch?v=TBCRlnwJtZU
'''

import os 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import logging 
from torch.utils.tensorboard import SummaryWriter
from modules_example import * 
from utils_example import *

# Build a class that performs the gaussian diffusion (forward process) 
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cpu'):
        '''
            Initialize the diffusion process parameters
        '''
        self.noise_steps = noise_steps # Number of steps in the diffusion process (MC length)
        self.beta_start = beta_start # Initial value of noising parameter
        self.beta_end = beta_end # Final value of noising parameter
        self.img_size = img_size # Input image size
        self.device = device # Device to run the diffusion process on

        self.beta = self.prepare_noise_schedule().to(device) # Noise schedule
        self.alpha = 1.0 - self.beta # Alpha param from DDPM paper
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # DDPM paper(shortcut to noise schedule)

    def prepare_noise_schedule(self):
        '''
            Prepare the noise schedule as a linear interpolation between beta_start and beta_end
        '''
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        '''
            Generate noise images for the diffusion process with a stochasticity in eps.
            Take in the input image(s) x and timestep(s) t
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps
    
    def sample_timesteps(self, n):
        '''
            Sample random timestep(s) for the diffusion process
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n):
        '''
            Sample n random images from the diffusion process
        '''
        logging.info("Sampling {n} new images....")
        model.eval()

        with torch.no_grad():
            '''
                This part follows Algo. 2 from DDPM paper
            '''
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device) # n rand imgs w 3 channels
            t = self.sample_timesteps(n).to(self.device) # Sampling random timesteps
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # Loop over the reversed range of noise steps
                t = (torch.ones(n) * i).long().to(self.device) # Set t to i
                predicted_noise = model(x, t) # Predict noise for x and t from model
                alpha = self.alpha[t][:, None, None, None] # Get alpha for t
                alpha_hat = self.alpha_hat[t][:, None, None, None] # Get alpha_hat for t
                beta = self.beta[t][:, None, None, None] # Get beta for t

                if i > 1:
                    noise = torch.randn_like(x) # Sample noise from N(0, 1)
                else:
                    noise = torch.zeros_like(x) # Set noise to 0 for first(last) step (reversed)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise # Update x with noise (DDPM algo 2)
        model.train()

        x = (x.clamp(-1, 1) + 1) / 2 # Clamp x to [-1, 1] and scale to [0, 1]
        # !!! clamping must be different for different atmospherical params

        return x 





def train(args):
    '''
        Training following Algo 1 from DDPM paper
    '''
    #setup_logging(args.run_name)
    device = args.device # Get device
    dataloader = get_data(args) # Get dataloader
    model = UNet().to(device) # Get model
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) # Get optimizer
    mse = nn.MSELoss() # Get loss function
    diffusion = Diffusion(img_size=args.img_size, device=device) # Get diffusion process
    logger = SummaryWriter(os.path.join("runs", args.run_name)) # Get logger
    l = len(dataloader) # Get length of dataloader


    for epoch in range(args.epochs):
        logging.info("Starting epoch {epoch}:")
        pbar = tqdm(dataloader) # Get progress bar
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device) # Get images in batch
            t = diffusion.sample_timesteps(images.shape[0]).to(device) # Sample random timesteps
            x_t, noise = diffusion.noise_images(images, t) # Get noised images and noise
            predicted_noise = model(x_t, t) # Predict noise for x_t and t
            loss = mse(noise, predicted_noise) # Calculate loss

            optimizer.zero_grad() # Zero gradients
            loss.backward() # Backpropagate
            optimizer.step() # Update weights

            pbar.set_postfix("MSE", loss.item()) # Update progress bar
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i) # Log loss

        sampled_images = diffusion.sample(model, n=images.shape[0]) # Sample images
        
        save_images(sampled_images, os.path.join("results", args.run_name, f"epoch_{epoch}.jpg")) # Save images
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt")) # Save model







def launch():
    import argparse 
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_unconditional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = "../data"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.lr = 1e-4
    train(args)

if __name__ == "__main__":
    launch()

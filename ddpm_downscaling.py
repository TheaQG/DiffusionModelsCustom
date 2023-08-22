import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO, datefmt='%I:%M:%S')

from modules_downscaling import *
from utils_downscaling import *




class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device='cpu'):
        '''
            Initialize the diffusion process parameters
        '''
        self.noise_steps = noise_steps # Number of steps in the diffusion process (MC length)
        self.beta_start = beta_start # Initial value of noising parameter
        self.beta_end = beta_end # Final value of noising parameter
        self.img_size = img_size # Input image size (n*n=img_size or img_size*img_size=n?)
        self.device = device # Device to run the diffusion process on

        self.beta = self.prepare_noise_schedule().to(device) # Noise schedule set to device
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
            !!!! This is also where conditionality is implemented in the future !!!!
            (in forward pass: t+= self.img_embedding(y_cond))
            For conditionality remember an embedding 
            
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
                Sample n random images from the diffusion process
            '''
            t = self.sample_timesteps(n).to(self.device) # Sample random timesteps
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device) # Sample random images with 3 channels (!!!! Channel number should be variable) 
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # Loop over timesteps in reverse order for backwards diffusion
                t = (torch.ones(n) * i).long().to(self.device) # Set timestep to i
                predicted_noise = model(x, t) # Predict noise for timestep i
                alpha =self.alpha[t][:, None, None, None] # Get alpha for timestep i
                alpha_hat = self.alpha_hat[t][:, None, None, None] # Get alpha_hat for timestep i
                beta = self.beta[t][:, None, None, None] # Get beta for timestep i

                if i > 1:
                    noise = torch.randn_like(x) # Sample noise for timestep i from N(0,1)
                else:
                    noise = torch.zeros_like(x) # Set noise to 0 for timestep 0, first (last) timestep (reversed)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise # Update x for timestep i with noise and predicted noise 

            model.train()

            x = (x.clamp(-1, 1) + 1) / 2 # Clamp x to [-1,1] and scale to [0,1]
            # Change clamping as atmospherical parameteres are changed. Scaling needs to be different

            return x #.cpu()
        

def train(args):
    '''
        Train the DDPM model as Algo 1 from paper
    '''

    setup_logging(args.run_name)
    device = args.device
    # !!!! Before running this, set up a function to get data !!!!
    dataLoader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device) 
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataLoader)


    for epoch in range(args.epoch):
        logging.info("Starting epoch {epoch}:")
        pbar = tqdm(dataLoader)

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix("MSE", loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])

        save_images(sampled_images, os.path.join("results", args.run_name, f"epoch_{epoch}.png"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))



def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_DS_unconditional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 256
    # !!!! Put in the right path for data !!!!
    args.dataset_path = "../data"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.lr = 1e-4
    train(args) 


if __name__ == "__main__":
    launch()







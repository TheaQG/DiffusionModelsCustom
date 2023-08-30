import os, tqdm, torch
from multiprocessing import freeze_support
import torch.nn as nn

from modules_DANRA_downscaling import *
from diffusion_DANRA_downscaling import DiffusionUtils




class TrainingPipeline:
    '''
        Class for building a training pipeline for DDPM.

    '''
    def __init__(self, model, lossfunc, optimizer, diffusion_utils:DiffusionUtils,
                 device='cpu', weight_init=True, custom_weight_initializer=None):
        '''
            Initialize the class.
            Input:
                - model: model to train
                - lossfunc: loss function to use
                - optimizer: optimizer to use
                - diffusion_utils: diffusion utils class
                - device: device to use (cpu or cuda)
                - weight_init: whether to initialize weights or not
                - custom_weight_initializer: custom weight initializer function
        '''

        # Set class variables
        self.device = device
        self.model = model.to(self.device)
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.diffusion_utils = diffusion_utils
        self.custom_weight_initializer = custom_weight_initializer

        # Initialize weights if weight_init is True
        if self.weight_init:
            if self.custom_weight_initializer:
                # Initialize weights with custom weight initializer
                self.model.apply(self.custom_weight_initializer)
            else:
                # Initialize weights with xavier uniform
                self.model.apply(self.xavier_init_weights)

    def xavier_init_weights(self, m):
        ''' 
            Function for initializing weights with xavier uniform.
            Xavier uniform is a good weight initializer for relu activation functions.
            It is not good for sigmoid or tanh activation functions.
            Input:
                - m: model to initialize weights for
        '''
        # Check if m is convolutional layer
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Initialize weights with xavier uniform
            nn.init.xavier_uniform_(m.weight)
            # If model has bias, initialize bias with constant 0.01
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    def save_model(self, dirname='./model_params', filename='DDPM.pth.tar'):
        '''
            Function for saving model parameters. 
            Input:
                - dirname: directory to save model parameters
                - filename: filename to save model parameters
        '''
        # Check if dirname exists, if not create it
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        
        # Set state dicts to save 
        state_dicts = {
            'network_params':self.model.state_dict(),
            'optimizer_params':self.optimizer.state_dict()
        }

        return torch.save(state_dicts, os.path.join(dirname, filename))
    

    def train(self, dataloader, verbose=True):
        '''
            Function for training model.
            Input:
                - dataloader: dataloader to use for training
                - verbose: whether to print training loss or not
        '''
        # Set model to train mode
        self.model.train()
        # Set loss to 0
        loss = 0

        # Loop over batches in dataloader, using tqdm for progress bar
        for idx, images in tqdm.tqdm(enumerate(dataloader)):
            # Set gradients to zero
            self.model.zero_grad()
            # Set images to device
            images = images.to(self.device)

            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)

            # Set predicted noise as output from model
            predicted_noise = self.model(x_t, t)

            # Calculate loss
            batch_loss = self.lossfunc(predicted_noise, noise)

            # Backpropagate loss
            batch_loss.backward()
            # Update weights
            self.optimizer.step()
            # Add batch loss to total loss
            loss += batch_loss.item()

        # Calculate average loss
        loss = loss / (idx + 1)

        # Print training loss if verbose is True
        if verbose:
            print(f'Training Loss: {loss}')

        return loss



if __name__ == '__main__':
    freeze_support()
import os, tqdm, torch
import torch.nn as nn
import matplotlib.pyplot as plt

from multiprocessing import freeze_support
from torch.cuda.amp import autocast, GradScaler

from ..src.utils import *
from ..src.data_modules import *
from ..src.diffusion_modules import DiffusionUtils


class TrainingPipeline_general:
    '''
        Class for building a training pipeline for DDPM.
        Possibility for various conditional inputs or none.

    '''
    def __init__(self,
                 model,
                 lossfunc,
                 optimizer,
                 diffusion_utils:DiffusionUtils,
                 device='cpu',
                 weight_init=True,
                 custom_weight_initializer=None,
                 sdf_weighted_loss=False
                 ):
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
        self.sdf_weighted_loss = sdf_weighted_loss

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
    

    def train(self,
              dataloader,
              verbose=True,
              PLOT_FIRST=False,
              SAVE_PATH='./',
              SAVE_NAME='upsampled_image.png',
              use_mixed_precision=True
              ):
        '''
            Function for training model.
            Input:
                - dataloader: dataloader to use for training
                - verbose: whether to print training loss or not
        '''
        # Set model to train mode
        self.model.train()
        # Set loss to 0
        loss = 0.0
        
        # Check if cuda is available and if mixed precision is available
        if torch.cuda.is_available() and use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Get information on variable for plotting
        var = dataloader.dataset.variable 

        # Set colormaps depending on variable
        if var == 'temp':
            cmap_name = 'plasma'
            cmap_label = 'Temperature [°C]'
        elif var == 'prcp':
            cmap_name = 'inferno'
            cmap_label = 'Precipitation [mm/day]'

        
        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, samples in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            # Samples is now a dict with available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples from dictionary
            images, seasons, cond_images, lsm, sdf, topo, _ = extract_samples(samples, self.device)
            
            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)
            
            
            # Set gradients to zero
            self.optimizer.zero_grad()

            # If mixed precision is available, use autocast
            if self.scaler:
                with autocast():
                    # Set predicted noise as output from model dependent on available conditions
                    predicted_noise = self.model(x_t,
                                                t,
                                                seasons,
                                                cond_images,
                                                lsm,
                                                topo)

                    # Calculate loss
                    if self.sdf_weighted_loss:
                        batch_loss = self.lossfunc(predicted_noise, noise, sdf)
                    else:
                        batch_loss = self.lossfunc(predicted_noise, noise)

                # Mixed precision: scale loss and backward pass
                self.scaler.scale(batch_loss).backward()
                # Update weights
                self.scaler.step(self.optimizer)
                # Update scaler
                self.scaler.update()

            # If mixed precision is not available, do not use autocast
            else:
                predicted_noise = self.model(x_t,
                                            t,
                                            seasons,
                                            cond_images,
                                            lsm,
                                            topo)

                # Calculate loss
                if self.sdf_weighted_loss:
                    batch_loss = self.lossfunc(predicted_noise, noise, sdf)
                else:
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
    
    def validate(self, dataloader, verbose=True, use_mixed_precision=True):
        # Set model to eval mode
        self.model.eval()

        # Set loss to 0
        val_loss = 0.0

        # Check if cuda is available and if mixed precision is available
        use_cuda = torch.cuda.is_available()

        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, samples in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            
            if 'img' in samples.keys():
                images = samples['img'].to(self.device)
            else:
                # Print error, you need to have images
                print('Error: you need to have images in your dataset')

            if 'classifier' in samples.keys():
                seasons = samples['classifier'].to(self.device)
            else:
                seasons = None        
            if 'img_cond' in samples.keys():
                # Check the data type of the samples
                cond_images = samples['img_cond'].to(self.device)
                cond_images = cond_images.to(torch.float)
            else:
                cond_images = None
            if 'lsm' in samples.keys():
                lsm = samples['lsm'].to(self.device)
            else:
                lsm = None
            if 'sdf' in samples.keys():
                sdf = samples['sdf'].to(self.device)
            else:
                sdf = None
            if 'topo' in samples.keys():
                topo = samples['topo'].to(self.device)
            else:
                topo = None

            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)


            # If mixed precision and CUDA is available, use autocast
            if use_cuda and use_mixed_precision:
                # Mixed precision training: autocast forward pass
                with autocast():

                    # Set predicted noise as output from model
                    predicted_noise = self.model(x_t, t, seasons, cond_images, lsm, topo)

                    # Calculate loss
                    if self.sdf_weighted_loss:
                        batch_loss = self.lossfunc(predicted_noise, noise, sdf)
                    else:
                        batch_loss = self.lossfunc(predicted_noise, noise)

            else:
                # Set predicted noise as output from model
                predicted_noise = self.model(x_t, t, seasons, cond_images, lsm, topo)

                # Calculate loss
                if self.sdf_weighted_loss:
                    batch_loss = self.lossfunc(predicted_noise, noise, sdf)
                else:
                    batch_loss = self.lossfunc(predicted_noise, noise)


            # Add batch loss to total loss
            val_loss += batch_loss.item()

        # Calculate average loss
        val_loss = val_loss / (idx + 1)

        # Print validation loss if verbose is True
        if verbose:
            print(f'Validation Loss: {val_loss}')

        return val_loss
import os, tqdm, torch

import torch.nn as nn
import matplotlib.pyplot as plt

from multiprocessing import freeze_support
from modules_DANRA_conditional import *
from diffusion_DANRA_conditional import DiffusionUtils
from torch.cuda.amp import autocast, GradScaler

class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.mse = nn.MSELoss()#nn.L1Loss()#

    def forward(self, predicted, target):
        return self.mse(predicted, target)

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, T=10):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.T = T
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        loss = self.mse(predictions[-1], targets[0])
        
        for t in range(1, self.T):
            loss += self.alpha * self.mse(predictions[t-1], targets[t])
        
        return loss

class SDFWeightedMSELoss(nn.Module):
    '''
        Custom loss function for SDFs.

    '''
    def __init__(self, max_land_weight=1.0, min_sea_weight=0.5):
        super().__init__()
        self.max_land_weight = max_land_weight
        self.min_sea_weight = min_sea_weight
#        self.mse = nn.MSELoss(reduction='none')

    def forward(self, input, target, sdf):
        # Convert SDF to weights, using a sigmoid function (or similar)
        # Scaling can be adjusted to control sharpness of transition
        weights = torch.sigmoid(sdf) * (self.max_land_weight - self.min_sea_weight) + self.min_sea_weight

        # Calculate the squared error
        squared_error = (input - target)**2

        # Apply the weights
        weighted_squared_error = weights * squared_error

        # Return mean of weighted squared error
        return weighted_squared_error.mean()

class TrainingPipeline_Hybrid:
    '''
        Class for building a training pipeline for DDPM.

    '''
    def __init__(self,
                 model,
                 lossfunc,
                 optimizer,
                 diffusion_utils:DiffusionUtils,
                 device='cpu',
                 weight_init=True,
                 custom_weight_initializer=None):
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
    

    def train(self, dataloader, verbose=True, PLOT_FIRST=False, SAVE_PATH='./', SAVE_NAME='upsampled_image.png'):
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
        
        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, (images, seasons) in tqdm.tqdm(enumerate(dataloader)):
            
            # Set gradients to zero
            self.model.zero_grad()
            # Set images and seasons to device
            images = images.to(self.device)
            seasons = seasons.to(self.device)
            
            if PLOT_FIRST:
                n_samples = 8
                images_samples = images[:n_samples]
                seasons_samples = seasons[:n_samples]



                fig, ax = plt.subplots(1, n_samples, figsize=(18, 4))
                for im, season, ax in zip(images_samples, seasons_samples, ax.flatten()):
                    image = ax.imshow(im.permute(1,2,0).cpu().detach().numpy())
                    ax.set_title(f'Season: {season.item()}')
                    #ax.axis('off')
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                
                fig.tight_layout()
                # plt.show()
                print(f'\n\n\nSaving {n_samples} upsampled training images...\n\n\n')
                fig.savefig(SAVE_PATH + '/' + SAVE_NAME, dpi=600, bbox_inches='tight')
                plt.close(fig)

                PLOT_FIRST = False

            # Sample timesteps fromƒdiffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)

            # Set predicted noise as output from model
            # Producing a sequence of predictions for different diffusion times
            predicted_noises = []
            for timestep in range(self.diffusion_utils.n_timesteps):
                predicted_noise_t = self.model(x_t, timestep, seasons)
                predicted_noises.append(predicted_noise_t)
            predicted_noises = torch.stack(predicted_noises)
            # Generating a sequence of noise for different diffusion times
            noise_sequence = []
            for timestep in range(self.diffusion_utils.n_timesteps):
                _, noise_t = self.diffusion_utils.noiseImage(images, timestep)
                noise_sequence.append(noise_t)
            noise_sequence = torch.stack(noise_sequence)

            # Calculate loss
            batch_loss = self.lossfunc(predicted_noises, noise_sequence)

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

        # Make sure all figures are closed
        plt.close('all')

        return loss





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
    

    def train(self, dataloader, verbose=True, PLOT_FIRST=False, SAVE_PATH='./', SAVE_NAME='upsampled_image.png'):
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
        
        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, (images, seasons) in tqdm.tqdm(enumerate(dataloader)):
            
            # Set gradients to zero
            self.model.zero_grad()
            # Set images and seasons to device
            images = images.to(self.device)
            
            from einops import rearrange
            images_re = rearrange(images, 'b c h w -> b c (h w)')
            ave_images = torch.mean(images_re, dim=2)
            
            cond_images = torch.ones_like(images)

            for n in range(len(images)):
                for c in range(len(images[n])):
                    cond_images[n,c,:,:] = ave_images[n,c]

            # print((cond_images[0,0,:,:]).shape)
            # print(cond_images[0,0,:,:])
            # print(cond_images.shape)

#            images = torch.cat((cond_images, images), dim=1)

            seasons = seasons.to(self.device)
            
            if PLOT_FIRST:
                n_samples = 8
                images_samples = images[:n_samples]
                seasons_samples = seasons[:n_samples]



                fig, ax = plt.subplots(1, n_samples, figsize=(18, 4))
                for im, season, ax in zip(images_samples, seasons_samples, ax.flatten()):
                    image = ax.imshow(im.permute(1,2,0).cpu().detach().numpy())
                    ax.set_title(f'Season: {season.item()}')
                    #ax.axis('off')
                    ax.set_ylim([0, im.shape[1]])
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                
                fig.tight_layout()
                #plt.show()
                print(f'\n\n\nSaving {n_samples} upsampled training images...\n\n\n')
                fig.savefig(SAVE_PATH + '/' + SAVE_NAME, dpi=600, bbox_inches='tight')
                plt.close(fig)

                PLOT_FIRST = False

            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)
            

            # Set predicted noise as output from model
            predicted_noise = self.model(x_t, t, seasons, cond_images)

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
        # Make sure all figures are closed
        plt.close('all')
        return loss
    
    def validate(self, dataloader, verbose=True):
        # Set model to eval mode
        self.model.eval()

        # Set loss to 0
        val_loss = 0.0

        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, (images, seasons) in tqdm.tqdm(enumerate(dataloader)):
            # Set images and seasons to device
            images = images.to(self.device)
            seasons = seasons.to(self.device)

            from einops import rearrange
            images_re = rearrange(images, 'b c h w -> b c (h w)')
            ave_images = torch.mean(images_re, dim=2)
            
            cond_images = torch.ones_like(images)

            for n in range(len(images)):
                for c in range(len(images[n])):
                    cond_images[n,c,:,:] = ave_images[n,c]

            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)

            # Set predicted noise as output from model
            predicted_noise = self.model(x_t, t, seasons, cond_images)

            # Calculate loss
            batch_loss = self.lossfunc(predicted_noise, noise)

            # Add batch loss to total loss
            val_loss += batch_loss.item()

        # Calculate average loss
        val_loss = val_loss / (idx + 1)

        # Print validation loss if verbose is True
        if verbose:
            print(f'Validation Loss: {val_loss}')

        # Make sure all figures are closed
        plt.close('all')

        return val_loss



class TrainingPipeline_new:
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
    

    def train(self, dataloader, verbose=True, PLOT_FIRST=False, SAVE_PATH='./', SAVE_NAME='upsampled_image.png'):
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
        
        # # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        # for idx, data in tqdm.tqdm(enumerate(dataloader)):
        #     # Investigate what shape the data is in (probably a mix of lists and tensors)
        #     print(data)

        for idx, samples in tqdm.tqdm(enumerate(dataloader)):
            #(images, seasons), (lsm, topo, point)
            (images, seasons), lsm, topo, points = samples
            
            # Set gradients to zero
            self.model.zero_grad()
            # Set images and seasons to device
            images = images.to(self.device)
            
            from einops import rearrange
            images_re = rearrange(images, 'b c h w -> b c (h w)')
            ave_images = torch.mean(images_re, dim=2)
            
            cond_images = torch.ones_like(images)

            for n in range(len(images)):
                for c in range(len(images[n])):
                    cond_images[n,c,:,:] = ave_images[n,c]

            # print((cond_images[0,0,:,:]).shape)
            # print(cond_images[0,0,:,:])
            # print(cond_images.shape)

#            images = torch.cat((cond_images, images), dim=1)

            seasons = seasons.to(self.device)
            
            if PLOT_FIRST:
                n_samples = 8
                images_samples = images[:n_samples]
                seasons_samples = seasons[:n_samples]
                cond_images_samples = cond_images[:n_samples]
                lsm_samples = lsm[:n_samples]
                topo_samples = topo[:n_samples]



                fig, axs = plt.subplots(1, n_samples, figsize=(18, 4))
                for im, season, ax in zip(images_samples, seasons_samples, axs.flatten()):
                    image = ax.imshow(im.permute(1,2,0).cpu().detach().numpy())
                    ax.set_title(f'Season: {season.item()}')
                    #ax.axis('off')
                    ax.set_ylim([0, im.shape[1]])
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                

                fig2, ax2s = plt.subplots(1, n_samples, figsize=(18, 4))
                for cond_im, season, ax in zip(cond_images_samples, seasons_samples, ax2s.flatten()):
                    image = ax.imshow(cond_im.permute(1,2,0).cpu().detach().numpy(), vmin = -10, vmax = 30)
                    ax.set_title(f'Season: {season.item()}')
                    #ax.axis('off')
                    ax.set_ylim([0, cond_im.shape[1]])
                    fig2.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig2.tight_layout()

                fig3, ax3s = plt.subplots(2, n_samples, figsize=(18, 4))
                i = 0
                for lsm_im, topo_im, season in zip(lsm_samples, topo_samples, seasons_samples):
                    image_lsm = ax3s[0, i].imshow(lsm_im.permute(1,2,0).cpu().detach().numpy())
                    ax3s[0, i].set_ylim([0, lsm_im.shape[1]])
                    image_topo = ax3s[1, i].imshow(topo_im.permute(1,2,0).cpu().detach().numpy())
                    ax3s[1, i].set_ylim([0, topo_im.shape[1]])
                    # ax.set_title(f'Season: {season.item()}')
                    # ax.axis('off')
                    # ax.set_ylim([0, im.shape[1]])
                    # fig3.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                    i += 1
                fig3.tight_layout()

                
                # for cond_im, season, ax in zip(images_samples, seasons_samples, ax.flatten()):
                #     image = ax.imshow(im.permute(1,2,0).cpu().detach().numpy())
                #     ax.set_title(f'Season: {season.item()}')
                #     ax.axis('off')
                #     ax.set_ylim([0, im.shape[1]])
                #     fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                
                fig.tight_layout()
                #plt.show()
                print(f'\n\n\nSaving {n_samples} upsampled training images...\n\n\n')
                fig.savefig(SAVE_PATH + '/' + SAVE_NAME, dpi=600, bbox_inches='tight')
                plt.close(fig)
                PLOT_FIRST = False

            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)

            # Set predicted noise as output from model
            predicted_noise = self.model(x_t, t, seasons, cond_images, lsm, topo)

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
        # Make sure all figures are closed
        plt.close('all')
        return loss
    
    def validate(self, dataloader, verbose=True):
        # Set model to eval mode
        self.model.eval()

        # Set loss to 0
        val_loss = 0.0

        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, samples in tqdm.tqdm(enumerate(dataloader)):
            (images, seasons), lsm, topo, points = samples
            # Set images and seasons to device
            images = images.to(self.device)
            seasons = seasons.to(self.device)
            lsm = lsm.to(self.device)
            topo = topo.to(self.device)
            

            from einops import rearrange
            images_re = rearrange(images, 'b c h w -> b c (h w)')
            ave_images = torch.mean(images_re, dim=2)
            
            cond_images = torch.ones_like(images)

            for n in range(len(images)):
                for c in range(len(images[n])):
                    cond_images[n,c,:,:] = ave_images[n,c]

            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)

            # Set predicted noise as output from model
            predicted_noise = self.model(x_t, t, seasons, cond_images, lsm, topo)

            # Calculate loss
            batch_loss = self.lossfunc(predicted_noise, noise)

            # Add batch loss to total loss
            val_loss += batch_loss.item()

        # Calculate average loss
        val_loss = val_loss / (idx + 1)

        # Print validation loss if verbose is True
        if verbose:
            print(f'Validation Loss: {val_loss}')

        # Make sure all figures are closed
        plt.close('all')

        return val_loss



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

        # Check if cuda is available and if mixed precision is avalilable
        if torch.cuda.is_available() and use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, samples in tqdm.tqdm(enumerate(dataloader)):
            # Samples is now a dict with available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Go through possible keys and send to device if available

            if 'img' in samples.keys():
                images = samples['img'].to(self.device)
                images = samples['img'].to(torch.float)
                # Print type of image tensor
                #print(f'Images are of type: {images.dtype}')
            else:
                # Print error, you need to have images
                print('Error: you need to have images in your dataset')

            if 'classifier' in samples.keys() and samples['classifier'] is not None:
                seasons = samples['classifier'].to(self.device)
                # Print type of seasons
                #print(f'Seasons are of type: {seasons.dtype}')
            else:
                seasons = None

            if 'img_cond' in samples.keys() and samples['img_cond'] is not None:
                # Check the data type of the samples
                cond_images = samples['img_cond'].to(self.device)
                cond_images = samples['img_cond'].to(torch.float)
                # Print type of cond_images
                #print(f'Conditional images are of type: {cond_images.dtype}')
            else:
                cond_images = None

            if 'lsm' in samples.keys() and samples['lsm'] is not None:
                lsm = samples['lsm'].to(self.device)
                lsm = samples['lsm'].to(torch.float)
                # Print type of lsm
                #print(f'LSM is of type: {lsm.dtype}')
            else:
                lsm = None

            if 'sdf' in samples.keys() and samples['sdf'] is not None:
                sdf = samples['sdf'].to(self.device)
                sdf = samples['sdf'].to(torch.float)
                # Print type of sdf
                #print(f'SDF is of type: {sdf.dtype}')
            else:
                sdf = None

            if 'topo' in samples.keys() and samples['topo'] is not None:
                topo = samples['topo'].to(self.device)
                topo = samples['topo'].to(torch.float)
                # Print type of topo
                #print(f'Topo is of type: {topo.dtype}')
            else:
                topo = None

            
            if PLOT_FIRST:
                n_samples = 8
                if 'img' in samples.keys():
                    images_samples = images[:n_samples]
                if 'classifier' in samples.keys():
                    seasons_samples = seasons[:n_samples]
                if 'img_cond' in samples.keys():
                    cond_images_samples = cond_images[:n_samples]
                if 'lsm' in samples.keys():
                    lsm_samples = lsm[:n_samples]
                if 'sdf' in samples.keys():
                    sdf_samples = sdf[:n_samples]
                if 'topo' in samples.keys():
                    topo_samples = topo[:n_samples]


                if 'img' in samples.keys():
                    fig, axs = plt.subplots(1, n_samples, figsize=(18, 4))
                    fig.suptitle('DANRA images')
                    i = 0
                    for im, ax in zip(images_samples, axs.flatten()):
                        image = ax.imshow(im.permute(1,2,0).cpu().detach().numpy())

                        if 'classifier' in samples.keys():
                            ax.set_title(f'Season: {seasons_samples[i].item()}')
                        else:
                            ax.set_title(f'Image {idx}')
                        #ax.axis('off')
                        ax.set_ylim([0, im.shape[1]])
                        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                        i += 1
                    fig.tight_layout()
                    

                if 'img_cond' in samples.keys():
                    fig2, ax2s = plt.subplots(1, n_samples, figsize=(18, 4))
                    fig2.suptitle('Conditional ERA5 images')
                    i = 0
                    for cond_im, ax in zip(cond_images_samples, ax2s.flatten()):
                        image = ax.imshow(cond_im.permute(1,2,0).cpu().detach().numpy())

                        if 'classifier' in samples.keys():
                            ax.set_title(f'Season: {seasons_samples[i].item()}')
                        else:
                            ax.set_title(f'Conditional image {idx}')
                        #ax.axis('off')
                        ax.set_ylim([0, cond_im.shape[1]])
                        fig2.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                        i += 1
                    fig2.tight_layout()
                    



                if 'sdf' or 'lsm' or 'topo' in samples.keys():
                    # Count how many of the keays are present and create new dict with only the present keys
                    n_keys = 0
                    geo_dict = {}
                    if 'lsm' in samples.keys():
                        n_keys += 1
                        geo_dict['lsm'] = lsm_samples
                    if 'topo' in samples.keys():
                        n_keys += 1
                        geo_dict['topo'] = topo_samples
                    if 'sdf' in samples.keys():
                        n_keys += 1
                        geo_dict['sdf'] = sdf_samples

                    if n_keys > 0:
                        fig3, ax3s = plt.subplots(n_keys, n_samples, figsize=(18, 6))

                        # Loop through keys and samples for each key
                        i = 0
                        for key, data in geo_dict.items():
                            for geo_sample, ax in zip(data, ax3s[i,:]):
                                image = ax.imshow(geo_sample.permute(1,2,0).cpu().detach().numpy())
                                
                                ax.axis('off')
                                ax.set_ylim([0, geo_sample.shape[1]])
                            i += 1
                        fig3.tight_layout()
                        fig3.savefig(SAVE_PATH + '/' + 'topo_lsm_' + SAVE_NAME, dpi=600, bbox_inches='tight')
                        plt.close(fig3)

                print(f'\n\n\nSaving {n_samples} upsampled training images...\n\n\n')
                if 'img' in samples.keys():
                    fig.savefig(SAVE_PATH + '/' + SAVE_NAME, dpi=600, bbox_inches='tight')
                    plt.close(fig)
                if 'img_cond' in samples.keys():
                    fig2.savefig(SAVE_PATH + '/' + 'cond_ERA5_' + SAVE_NAME, dpi=600, bbox_inches='tight')
                    plt.close(fig2)

                #plt.show()

                PLOT_FIRST = False

            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)


            # Set gradients to zero
            self.optimizer.zero_grad()

            # If mixed prcision is available, use autocast
            if self.scaler:
                # Set mixed precision training
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
            else:
                predicted_noise = self.model(x_t,
                                             t,
                                             seasons,
                                             cond_images,
                                             lsm,
                                             topo
                                             )
                
                # Calculate loss
                if self.sdf_weighted_loss:
                    batch_loss = self.lossfunc(predicted_noise, noise, sdf)
                else:
                    batch_loss = self.lossfunc(predicted_noise, noise)
                
                # Backpropagate loss
                batch_loss.backward()
                # Update weights
                self.optimizer.step()


            loss += batch_loss.item()

        # Calculate average loss
        loss = loss / (idx + 1)

        # Print training loss if verbose is True
        if verbose:
            print(f'Training Loss: {loss}')

        # Make sure all figures are closed
        plt.close('all')
        return loss
    
    def validate(self,
                 dataloader,
                 verbose=True,
                 use_mixed_precision=True
                 ):
        # Set model to eval mode
        self.model.eval()

        # Set loss to 0
        val_loss = 0.0

        # Check if cuda is available
        use_cuda = torch.cuda.is_available()

        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, samples in tqdm.tqdm(enumerate(dataloader)):
            
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

            # If mixed precision and CUDA available, use autocast
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


class TrainingPipeline_ERA5_Condition:
    '''
        Class for building a training pipeline for DDPM.

    '''
    def __init__(self,
                 model,
                 lossfunc,
                 optimizer,
                 diffusion_utils:DiffusionUtils,
                 device='cpu',
                 weight_init=True,
                 custom_weight_initializer=None,
                 sdf_weighted_loss = False):
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
              SAVE_NAME='upsampled_image.png'):
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
        
        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, samples in tqdm.tqdm(enumerate(dataloader)):
            
            if self.sdf_weighted_loss:
                (images, seasons, cond_images), lsm, topo, sdf, _ = samples    
            else:
                (images, seasons, cond_images), lsm, topo, _ = samples
            # Check the data type of the samples
            cond_images = cond_images.to(torch.float)

            # Set gradients to zero
            self.model.zero_grad()
            # Set images and seasons to device
            images = images.to(self.device)
            cond_images = cond_images.to(self.device)
            seasons = seasons.to(self.device)
            lsm = lsm.to(self.device)
            topo = topo.to(self.device)
            if self.sdf_weighted_loss:
                sdf.to(self.device)

            
            
            if PLOT_FIRST:
                n_samples = 8
                images_samples = images[:n_samples]
                seasons_samples = seasons[:n_samples]
                cond_images_samples = cond_images[:n_samples]
                lsm_samples = lsm[:n_samples]
                topo_samples = topo[:n_samples]



                fig, axs = plt.subplots(1, n_samples, figsize=(18, 4))
                fig.suptitle('DANRA images')
                for im, season, ax in zip(images_samples, seasons_samples, axs.flatten()):
                    image = ax.imshow(im.permute(1,2,0).cpu().detach().numpy())
                    ax.set_title(f'Season: {season.item()}')
                    #ax.axis('off')
                    ax.set_ylim([0, im.shape[1]])
                    #fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()

                fig2, ax2s = plt.subplots(1, n_samples, figsize=(18, 4))
                fig2.suptitle('Conditional ERA5 images')
                for cond_im, season, ax in zip(cond_images_samples, seasons_samples, ax2s.flatten()):
                    image = ax.imshow(cond_im.permute(1,2,0).cpu().detach().numpy())
                    ax.set_title(f'Season: {season.item()}')
                    #ax.axis('off')
                    ax.set_ylim([0, cond_im.shape[1]])
                    fig2.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig2.tight_layout()

                fig3, ax3s = plt.subplots(2, n_samples, figsize=(18, 4))
                fig3.suptitle('Topography and land/sea mask')
                i = 0
                for lsm_im, topo_im, season in zip(lsm_samples, topo_samples, seasons_samples):
                    ax3s[0, i].imshow(lsm_im.permute(1,2,0).cpu().detach().numpy())
                    ax3s[0, i].set_ylim([0, lsm_im.shape[1]])
                    ax3s[1, i].imshow(topo_im.permute(1,2,0).cpu().detach().numpy())
                    ax3s[1, i].set_ylim([0, topo_im.shape[1]])
                    i += 1
                fig3.tight_layout()

                
                #plt.show()
                print(f'\n\n\nSaving {n_samples} upsampled training images...\n\n\n')
                fig.savefig(SAVE_PATH + '/' + SAVE_NAME, dpi=600, bbox_inches='tight')
                fig2.savefig(SAVE_PATH + '/' + 'cond_ERA5_' + SAVE_NAME, dpi=600, bbox_inches='tight')
                fig3.savefig(SAVE_PATH + '/' + 'topo_lsm_' + SAVE_NAME, dpi=600, bbox_inches='tight')
                plt.close(fig)
                plt.close(fig2)
                plt.close(fig3)
                PLOT_FIRST = False

            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)

            # Set predicted noise as output from model
            predicted_noise = self.model(x_t, t, seasons, cond_images, lsm, topo)

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

        # Make sure all figures are closed
        plt.close('all')
        
        return loss
    
    def validate(self, dataloader, verbose=True):
        # Set model to eval mode
        self.model.eval()

        # Set loss to 0
        val_loss = 0.0

        # Loop over batches(tuple of img and seasons) in dataloader, using tqdm for progress bar
        for idx, samples in tqdm.tqdm(enumerate(dataloader)):
            if self.sdf_weighted_loss:
                (images, seasons, cond_images), lsm, topo, sdf, _ = samples
            else:
                (images, seasons, cond_images), lsm, topo, _ = samples
            
            # Set images and seasons to device
            images = images.to(self.device)
            seasons = seasons.to(self.device)
            lsm = lsm.to(self.device)
            topo = topo.to(self.device)
            cond_images = cond_images.to(torch.float)
            cond_images = cond_images.to(self.device)
            
            # Sample timesteps from diffusion utils
            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            # Sample noise from diffusion utils
            x_t, noise = self.diffusion_utils.noiseImage(images, t)

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




if __name__ == '__main__':
    freeze_support()
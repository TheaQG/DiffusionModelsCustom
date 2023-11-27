import os, tqdm, torch
from multiprocessing import freeze_support
import torch.nn as nn
import torch.nn.functional as F

from modules_DANRA_conditional import *
from diffusion_DANRA_conditional import DiffusionUtils

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


class TrainingPipeline_Hybrid:
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
                    ax.axis('off')
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                
                fig.tight_layout()
                plt.show()
                print(f'\n\n\nSaving {n_samples} upsampled training images...\n\n\n')
                fig.savefig(SAVE_PATH + '/' + SAVE_NAME, dpi=600, bbox_inches='tight')
                plt.close(fig)

                PLOT_FIRST = False

            # Sample timesteps from diffusion utils
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
                    ax.axis('off')
                    ax.set_ylim([0, im.shape[1]])
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                
                fig.tight_layout()
                plt.show()
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
                    ax.axis('off')
                    ax.set_ylim([0, im.shape[1]])
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                

                fig2, ax2s = plt.subplots(1, n_samples, figsize=(18, 4))
                for cond_im, season, ax in zip(cond_images_samples, seasons_samples, ax2s.flatten()):
                    image = ax.imshow(cond_im.permute(1,2,0).cpu().detach().numpy(), vmin = -10, vmax = 30)
                    ax.set_title(f'Season: {season.item()}')
                    ax.axis('off')
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
                plt.show()
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

        return val_loss




if __name__ == '__main__':
    freeze_support()
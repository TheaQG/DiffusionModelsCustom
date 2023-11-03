'''
    Script to perform training of DDPM on DANRA data, with conditional training on season labels.
    With validation set and saving of model checkpoints.

    Based on https://github.com/dome272/Diffusion-Models-pytorch 
    and
    https://arxiv.org/abs/2006.11239 (DDPM)
'''

import os, tqdm, random, torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Union, Iterable, Tuple


# Import objects from other files in this repository
from data_DANRA_conditional import DANRA_Dataset, preprocess_lsm_topography
from modules_DANRA_conditional import *
from diffusion_DANRA_conditional import DiffusionUtils
from training_DANRA_conditional import *

# Run 'export MKL_SERVICE_FORCE_INTEL=1' in bash

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


if __name__ == '__main__':
    print('\n\n')
    print('#'*50)
    print('Running ddpm_DANRA_conditional.py')
    print('#'*50)
    print('\n\n')

    from multiprocessing import freeze_support
    freeze_support()

    # General settings for use

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

    # Define data hyperparameters
    input_channels = 1
    output_channels = 1
    n_samples = n_files
    cache_size = n_files
    image_dim = 32#64#n_danra_size#
    image_size = (image_dim,image_dim)
    n_seasons = 4#12#366#
    loss_type = 'simple'#'hybrid'#
    p_train = 0.8 # Train split
    # p_valid = 0.2 # Validation split 
    # p_test = 0.0 # Test split

    # Define strings for use in path
    im_dim_str = str(image_dim) + 'x' + str(image_dim)
    cond_str = 'lsm_topo__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_wValid_600Epochs_batch32'
    var_str = var
    model_str = 'DDPM_conditional_wValid_600Epochs_batch32'
    # Set path to save figures
    SAVE_FIGS = False
    PATH_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures'
    PATH_SAMPLES = PATH_SAVE + f'/Samples/Samples' + '__' + var_str + '__' + im_dim_str + '__' + cond_str 
    PATH_LOSSES = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses'
    if not os.path.exists(PATH_SAMPLES):
        os.makedirs(PATH_SAMPLES)
    
    NAME_SAMPLES = 'Generated_samples' + '__' + var_str + '__' + im_dim_str + '__' + cond_str + '__' + 'epoch' + '_'
    NAME_FINAL_SAMPLES = f'Final_generated_sample' + '__' + var_str + '__' + im_dim_str + '__' + cond_str 
    NAME_LOSSES = f'Training_losses' + '__' + var_str + '__' + im_dim_str + '__' + cond_str


    # Define the path to the pretrained model 
    PATH_CHECKPOINT = '/Users/au728490/Documents/PhD_AU/Python_Scripts/ModelCheckpoints/DDPM_DANRA/'
    try:
        os.makedirs(PATH_CHECKPOINT)
        print('\n\n\nCreating directory for saving checkpoints...')
        print(f'Directory created at {PATH_CHECKPOINT}')
    except FileExistsError:
        print('\n\n\nDirectory for saving checkpoints already exists...')
        print(f'Directory located at {PATH_CHECKPOINT}')
    

    NAME_CHECKPOINT = model_str + '__' + var_str + '__' + im_dim_str + '__' + cond_str + '.pth.tar'

    checkpoint_dir = PATH_CHECKPOINT
    checkpoint_name = NAME_CHECKPOINT
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f'\nCheckpoint path: {checkpoint_path}')


    # Test model input<->output before running?
    TEST_MODEL = True

    PLOT_EPOCH_SAMPLES = True


    # Define model hyperparameters
    epochs = 600
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    first_fmap_channels = 64#n_danra_size #
    last_fmap_channels = 512 #2048
    time_embedding = 256
    learning_rate = 3e-4 #1e-2
    min_lr = 1e-6
    weight_decay = 0.0

    # Define diffusion hyperparameters
    n_timesteps = 800
    beta_min = 1e-4
    beta_max = 0.02
    beta_scheduler = 'linear'

    # Preprocess lsm and topography (normalize and reshape, resizes to image_size)
    PATH_LSM = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_DK/lsm_dk.npz'
    PATH_TOPO = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_DK/topo_dk.npz'
    lsm_tensor, topo_tensor = preprocess_lsm_topography(PATH_LSM, PATH_TOPO, image_size, scale=False)#, scale=True)
    lsm_weight = 1
    lsm_tensor = lsm_weight * lsm_tensor#None#
    topo_weight = 1
    topo_tensor = topo_weight * topo_tensor#None#
    
    n_plots = 0
    
    # Print shape of lsm and topography tensors
    if lsm_tensor is not None:
        print(f'\n\n\nShape of lsm tensor: {lsm_tensor.shape}')
        n_plots += 1
    if topo_tensor is not None:
        print(f'Shape of topography tensor: {topo_tensor.shape}\n\n')
        n_plots += 1
    
    # Plot lsm and topography tensors
    
    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        if lsm_tensor is not None:
            ax.imshow(lsm_tensor.squeeze(), cmap='viridis')
            ax.set_title(f'Land-sea mask, upscaled to {image_size[0]}x{image_size[1]}')
        elif topo_tensor is not None:
            ax.imshow(topo_tensor.squeeze(), cmap='viridis')
            ax.set_title(f'Topography, upscaled to {image_size[0]}x{image_size[1]}')
        fig.savefig(PATH_SAMPLES + f'/ddpm_conditional__{var}_lsm_topo.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

    elif n_plots == 2:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(lsm_tensor.squeeze(), cmap='viridis')
        axs[0].set_title(f'Land-sea mask, upscaled to {image_size[0]}x{image_size[1]}')
        axs[1].imshow(topo_tensor.squeeze(), cmap='viridis')
        axs[1].set_title(f'Topography, upscaled to {image_size[0]}x{image_size[1]}')
        print(f'\n\n\nSaving lsm and topography figure...')
        print('\n\n')
        fig.savefig(PATH_SAMPLES + f'/ddpm_conditional__{var}_lsm_topo.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

    elif n_plots == 0:
        print(f'\n\n\nNo lsm or topography tensor found...\n\n')
        print(f'Continuing without lsm and topography tensors...\n\n')

    


    # Define the dataset from data_DANRA_downscaling.py
    train_dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size, scale=False, conditional=True, n_classes=n_seasons)

    # Split the dataset into train and validation
    train_size = int(p_train * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    # Define the torch dataloaders for train and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Define the seed for reproducibility, and set seed for torch, numpy and random
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set torch to deterministic mode, meaning that the same input will always produce the same output
    torch.backends.cudnn.deterministic = False
    # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
    torch.backends.cudnn.benchmark = True

    # Define the encoder and decoder from modules_DANRA_downscaling.py
    encoder = Encoder(input_channels, time_embedding, lsm_tensor=lsm_tensor, topo_tensor=topo_tensor, block_layers=[2, 2, 2, 2], num_classes=n_seasons)
    decoder = Decoder(last_fmap_channels, output_channels, time_embedding, first_fmap_channels)
    # Define the model from modules_DANRA_downscaling.py
    model = DiffusionNet(encoder, decoder)
    # Define the diffusion utils from diffusion_DANRA_downscaling.py
    diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, device, beta_scheduler)
    # Define the loss function

    if loss_type == 'simple':
        lossfunc = SimpleLoss()
    elif loss_type == 'hybrid':
        lossfunc = HybridLoss(alpha=0.5, T=n_timesteps)#nn.MSELoss()#SimpleLoss()#
    
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Define the training pipeline from training_DANRA_downscaling.py
    if loss_type == 'simple':
        pipeline = TrainingPipeline(model, lossfunc, optimizer, diffusion_utils, device, weight_init=True)
    elif loss_type == 'hybrid':
        pipeline = TrainingPipeline_Hybrid(model, lossfunc, optimizer, diffusion_utils, device, weight_init=True)
    
    # Define the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer, 'min', factor=0.5, patience=5, verbose=True, min_lr=min_lr)
    #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(pipeline.optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=len(train_dataloader), pct_start=0.3, anneal_strategy='cos', final_div_factor=300)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, epochs=N, steps_per_epoch=len(data_loader), pct_start=0.3, anneal_strategy='cos', final_div_factor=300)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)    

    # Check if the path to pretrained model exists
    if os.path.isfile(checkpoint_path):
        # If the path exists, load the pretrained weights
        print('\n\n\nLoading pretrained weights...')
        print(checkpoint_path)
        checkpoint_state = torch.load(checkpoint_path, map_location=device)['network_params']
        pipeline.model.load_state_dict(checkpoint_state)

    if TEST_MODEL:
        print('\n\n\nTesting model output size...\n')
        n_samples = 4

        test_input = torch.randn(n_samples, input_channels, *image_size).to(device)
        t_test = torch.randint(0, time_embedding, (n_samples,)).to(device)#torch.Tensor([1, 5, 7, 11]).to(device)#
        y_test = torch.randint(0, n_seasons, (n_samples, ))#torch.zeros(n_samples)
        #idx_test = np.random.randint(0, n_seasons)
        #y_test[idx_test] = 1
        y_test = y_test.type(torch.int64).to(device)

        print(f'Input shape of test: {test_input.shape}')
        print(f'Shape of time embedding: {t_test.shape}')
        print(f'Shape of season embedding: {y_test.shape}')
        test = pipeline.model(test_input, t_test, y_test)

        
        print(f'Output shape of test: {test.shape}')
        print('\n')
        print('Number of parameters: ' + str(sum([i.numel() for i in pipeline.model.parameters()])))
        print('\n')

    # Check if device is cuda, if so print information and empty cache
    if torch.cuda.is_available():
        print(f'Model is training on {torch.cuda.get_device_name()}\n\n')
        print(f'Model is using {torch.cuda.memory_allocated()} bytes of memory\n\n')
        torch.cuda.empty_cache()

    # Set empty list for train losses
    train_losses = []
    valid_losses = []

    # Set best loss to infinity
    best_loss = np.inf

    print(f'\n\n\nStarting training...\n\n')
    # Loop over epochs
    for epoch in range(epochs):
        
        print(f'\n\nEpoch {epoch+1} of {epochs}:\n\n')
        if epoch == 0:
            PLOT_FIRST_IMG = True
        else:
            PLOT_FIRST_IMG = False

        # Call the train function from pipeline
        train_loss = pipeline.train(train_dataloader, verbose=False, PLOT_FIRST=PLOT_FIRST_IMG, SAVE_PATH=PATH_SAMPLES, SAVE_NAME='upsampled_images_example.png')
        # Append train loss to list
        train_losses.append(train_loss)
        
        # Call the evaluate function from pipeline
        valid_loss = pipeline.validate(valid_dataloader, verbose=False)
        # Append valid loss to list
        valid_losses.append(valid_loss)

        # # Print train and valid loss
        if valid_loss > best_loss:
            print(f'\n\nTraining Loss: {train_loss:.6f}\n\n')
            print(f'Validation Loss: {valid_loss:.6f}\n\n')

        # If valid loss is better than best loss, save model
        if valid_loss < best_loss:
            best_loss = valid_loss
            pipeline.save_model(checkpoint_dir, checkpoint_name)
            print(f'Model saved at epoch {epoch+1} with validation loss {best_loss:.6f}')
            print(f'Training loss: {train_loss:.6f}')
            print(f'Saved to {checkpoint_dir} with name {checkpoint_name}')


        # # If train loss is better than best loss, save model
        # if train_loss < best_loss:
        #     best_loss = train_loss
        #     pipeline.save_model(checkpoint_dir, checkpoint_name)
        #     print(f'Model saved at epoch {epoch+1} with loss {best_loss}')
        #     print(f'Saved to {checkpoint_dir} with name {checkpoint_name}')

        # If epoch is multiple of 10 or last epoch, generate samples
        if (epoch == 0 and PLOT_EPOCH_SAMPLES) or (((epoch + 1) % 10) == 0 and PLOT_EPOCH_SAMPLES) or (epoch == (epochs - 1) and PLOT_EPOCH_SAMPLES):
            print('Generating samples...')
            n = 4
            # Generate random fields of batchsize n
            x = torch.randn(n, input_channels, *image_size).to(device)

            # Generate random season labels of batchsize n
            y = torch.randint(0, n_seasons, (n,)).to(device) # 4 seasons, 0-3

            # Sample generated images from model
            generated_images = diffusion_utils.sample(x, pipeline.model, y)
            generated_images = generated_images.detach().cpu()

            # Plot generated images
            fig, axs = plt.subplots(1, n, figsize=(8,3))

            for i in range(n):
                img = generated_images[i].squeeze()
                season = y[i].item()
                image = axs[i].imshow(img, cmap='viridis')
                fig.colorbar(image, ax=axs[i], fraction=0.046, pad=0.04)
                axs[i].set_title(f'Season: {y[i].item()}')
                axs[i].axis('off')

            fig.tight_layout()
            fig.savefig(PATH_SAMPLES + '/' + NAME_SAMPLES + str(epoch+1) + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)


            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(train_losses, label='Train')
            ax.plot(valid_losses, label='Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss')
            ax.legend(loc='upper right')
            fig.savefig(PATH_SAMPLES + '/' + NAME_LOSSES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

            with open(PATH_LOSSES + '/' + NAME_LOSSES + '_train', 'wb') as fp:
                pickle.dump(train_losses, fp)
            with open(PATH_LOSSES + '/' + NAME_LOSSES + '_valid', 'wb') as fp:
                pickle.dump(valid_losses, fp)

        # Step the learning rate scheduler
        lr_scheduler.step(train_loss)

    # Load best model state
    best_model_path = checkpoint_path#os.path.join('../../ModelCheckpoints/DDPM_DANRA', 'DDPM.pth.tar')
    best_model_state = torch.load(best_model_path)['network_params']

    # Load best model state into model
    pipeline.model.load_state_dict(best_model_state)

    print('Generating samples...')

    # Set number of samples to generate
    
    n = 8

    fig, ax = plt.subplots(n_seasons, n, figsize=(18, 10))

    if n_seasons < 13:
        for season in range(n_seasons):
            # Generate random fields of batchsize n
            x = torch.randn(n, input_channels, *image_size).to(device)
            # Generate season labels of batchsize n (generating n fields for each season)
            y = torch.ones(n) * season
            # Set season labels to device as int64
            y = y.type(torch.int64).to(device)

            # Sample generated images from model
            generated_images = diffusion_utils.sample(x, pipeline.model, y)
            generated_images = generated_images.detach().cpu()

            for i in range(n):
                img = generated_images[i].squeeze()
                image = ax[season, i].imshow(img, cmap='viridis')
                ax[season, i].set_title(f'Season: {y[i].item()}')
                ax[season, i].axis('off')
                fig.colorbar(image, ax=ax[season, i], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(PATH_SAMPLES + '/' + NAME_FINAL_SAMPLES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    else:
        n_sampling = 6
        n_seasons_random = random.randint(0, n_seasons, n_sampling)
#        n_seasons_random = random.sample(range(n_seasons), n_sampling)
        for season in range(n_seasons_random):
            # Generate random fields of batchsize n
            x = torch.randn(n, input_channels, *image_size).to(device)
            # Generate season labels of batchsize n (generating n fields for each season)
            y = torch.ones(n) * season
            # Set season labels to device as int64
            y = y.type(torch.int64).to(device)

            # Sample generated images from model
            generated_images = diffusion_utils.sample(x, pipeline.model, y)
            generated_images = generated_images.detach().cpu()

            for i in range(n):
                img = generated_images[i].squeeze()
                image = ax[season, i].imshow(img, cmap='viridis')
                ax[season, i].set_title(f'Season: {y[i].item()}')
                ax[season, i].axis('off')
                fig.colorbar(image, ax=ax[season, i], fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(PATH_SAMPLES + '/' + NAME_FINAL_SAMPLES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.show()









    # # Set number of samples to generate
    # n = 4

    # # Generate samples
    # x = torch.randn(n, input_channels, *image_size).to(device)

    # # Generate random season labels of batchsize n
    # y = torch.randint(0, 4, (n,)).to(device) # 4 seasons, 0-3

    # # Sample generated images from model
    # generated_images = diffusion_utils.sample(x, pipeline.model, y)
    # generated_images = generated_images.detach().cpu()

    # # Plot samples
    # fig, axs = plt.subplots(1, n, figsize=(8,3))

    # for i in range(n):
    #     img = generated_images[i].squeeze()
    #     image = axs[i].imshow(img, cmap='viridis')
    #     axs[i].set_title(f'Season: {y[i].item()}')
    #     axs[i].axis('off')

    # fig.tight_layout()
    # fig.savefig(PATH_SAMPLES + '/' + NAME_FINAL_SAMPLES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    # plt.show()






# if __name__ == '__main__':

    

#     sample_img = dataset[0]
#     print(type(sample_img))
#     print('\n')
#     print(f'shape: {sample_img.shape}')
#     print(f'min pixel value: {sample_img.min()}')
#     print(f'mean pixel value: {sample_img.mean()}')
#     print(f'max pixel value: {sample_img.max()}')


#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ax.set_title('Sample Image, ' + var)
#     img = sample_img.squeeze()#
#     image = ax.imshow(img, cmap='viridis')
#     fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    
#     plt.show()

#     T = n_timesteps
#     n_steps = 50
#     alpha_values = {}

#     # print('\n')
#     # for scheduler in ['linear', 'cosine']:
#     #     print(f'Running {scheduler} beta scheduler...')
        
#     #     diffusion = DiffusionUtils(T, beta_min, beta_max, scheduler=scheduler)
#     #     alpha_values[scheduler] = diffusion.alphas

#     #     fig, axs = plt.subplots(2, (T//(n_steps*2))+1, figsize=(15,8))
        
#     #     axs.flatten()[0].imshow(sample_img.squeeze())
#     #     axs.flatten()[0].set_title('Original Image')

#     #     for idx, t in enumerate(range(n_steps-1, T, n_steps)):
#     #         t = torch.Tensor([t]).long()
#     #         x, _ = diffusion.noiseImage(sample_img.unsqueeze(0), t)
#     #         axs.flatten()[idx+1].imshow(x.squeeze())
#     #         axs.flatten()[idx+1].set_title(f'Image at t={t.item()}')

#     #     fig.set_tight_layout(True)
#     #     if SAVE_FIGS:
#     #         fig.savefig(PATH_SAVE + f'/ddpm_kaggle_ex__{scheduler}_diffusion.png')

#     #     print('\n')



#     # fig, axs = plt.subplots(1, 2, figsize=(20, 4))

#     # axs[0].plot(alpha_values['linear'])
#     # axs[0].set_xlabel('timestep (t)')
#     # axs[0].set_ylabel('alpha (1-beta)')
#     # axs[0].set_title('alpha values of linear scheduling')

#     # axs[1].plot(alpha_values['cosine'])
#     # axs[1].set_xlabel('timestep (t)')
#     # axs[1].set_ylabel('alpha (1-beta)')
#     # axs[1].set_title('alpha values of cosine scheduling')

#     # plt.show()



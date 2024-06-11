'''
    Need to implement:
    - When loading pretrained model, check what epoch it is from. If it is from a previous epoch, continue training from that epoch and save figs and losses continued from that epoch
    
'''

import os, torch
import pickle
import zarr
import argparse

import numpy as np
import torch.distributed as dist

from multiprocessing import freeze_support
from torchinfo import summary
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Import objects from other files in this repository
from .data_modules import DANRA_Dataset_cutouts_ERA5_Zarr, preprocess_lsm_topography
from .unet import *
from .diffusion_modules import DiffusionUtils
from .training import *
from .utils import *

def make_zarr_data_from_all_data(data_dir, var, size_str, data_split_type, data_split_params, scaling):
    '''
        Function to make zarr data from all data in the data directory.
        Saves the zarr data to the zarr_files directory.
    '''
    # Set path to zarr files
    zarr_dir = data_dir + '/zarr_files/'
    try:
        os.makedirs(zarr_dir)
    except FileExistsError:
        print('Directory already exists')
    # Set path to save zarr data
    zarr_path = zarr_dir + var + '_' + size_str + '_' + data_split_type + '.zarr'
    # Load the data
    data = np.load(data_dir + '/data.npz')
    # Set the data
    data = data['data']
    # Set the data split
    data_split = data_split_params['data_split']
    # Set the data split type
    data_split_type = data_split_params['data_split_type']
    # Set the data split params
    data_split_params = data_split_params['data_split_params']
    


def main_ddpm(args):
    
    print('\n\n')
    print('#'*50)
    print('Running ddpm')
    print('#'*50)
    print('\n\n')


    # General settings for use

    # Define DANRA data information 
    # Set variable for use
    var = args.HR_VAR 
    lr_vars = args.LR_VARS

    # Set scaling to true or false
    scaling = args.scaling

    # Set some color map settings
    if var == 'temp':
        cmap_name = 'plasma'
        cmap_label = 'Temperature [°C]'
    elif var == 'prcp':
        cmap_name = 'inferno'
        cmap_label = 'Precipitation [mm]'

    # Set DANRA size string for use in path
    danra_size_str = '589x789'

    PATH_SAVE = args.path_save
    
    # Path: .../Data_DiffMod
    # To HR data: Path + '/data_DANRA/size_589x789/' + var + '_' + danra_size_str +  '/zarr_files/train.zarr'
    PATH_HR = args.path_data + 'data_DANRA/size_589x789/' + var + '_' + danra_size_str +  '/zarr_files/'
    # Path to DANRA data (zarr files), full danra, to enable cutouts
    print(PATH_HR)
    data_dir_danra_train_w_cutouts_zarr = PATH_HR + 'train.zarr'
    data_dir_danra_valid_w_cutouts_zarr = PATH_HR + 'valid.zarr'
    data_dir_danra_test_w_cutouts_zarr = PATH_HR + 'test.zarr'

    # To LR data: Path + '/data_ERA5/size_589x789/' + var + '_' + danra_size_str +  '/'
    # Path to ERA5 data, 589x789 (same size as DANRA)
    print(lr_vars)
    if args.LR_VARS is not None:
        print(len(lr_vars))
        if len(lr_vars) == 1:
            lr_var = lr_vars[0]
            PATH_LR = args.path_data + 'data_ERA5/size_589x789/' + lr_var + '_' + danra_size_str +  '/zarr_files/'
            data_dir_era5_train_zarr = PATH_LR + 'train.zarr'
            data_dir_era5_valid_zarr = PATH_LR + 'valid.zarr'
            data_dir_era5_test_zarr = PATH_LR + 'test.zarr'
        if len(lr_vars) > 1:
            # NEED TO IMPLEMENT CONCATENATION OF MULTIPLE VARIABLES (before training, and save to zarr file)
            KeyError('Multiple variables not yet implemented')
            # Check if .zarr files in 'concat_' str_lr_vars + '_' + danra_size_str + '/zarr_files/' exists
            str_lr_vars = '_'.join(lr_vars)
            PATH_LR = args.path_data + 'data_ERA5/size_589x789' + 'concat_' + str_lr_vars + '_' + danra_size_str + '/zarr_files/' 
            data_dir_era5_train_zarr = PATH_LR + 'train.zarr'
            # Check if .zarr file exists and otherwise raise error
            if not os.path.exists(data_dir_era5_train_zarr):
                raise FileNotFoundError(f'File not found: {data_dir_era5_train_zarr}. If multiple variables are used, the zarr file should be concatenated and saved in the directory: {PATH_LR}')
            
            data_dir_era5_valid_zarr = PATH_LR + 'valid.zarr'
            data_dir_era5_test_zarr = PATH_LR + 'test.zarr'
    else:
        data_dir_era5_train_zarr = None
        data_dir_era5_valid_zarr = None
        data_dir_era5_test_zarr = None
        
    # Make zarr groups
    data_danra_train_zarr = zarr.open_group(data_dir_danra_train_w_cutouts_zarr, mode='r')
    data_danra_valid_zarr = zarr.open_group(data_dir_danra_valid_w_cutouts_zarr, mode='r')
    data_danra_test_zarr = zarr.open_group(data_dir_danra_test_w_cutouts_zarr, mode='r')

# /scratch/project_465000956/quistgaa/Data/Data_DiffMod/
    n_files_train = len(list(data_danra_train_zarr.keys()))
    n_files_valid = len(list(data_danra_valid_zarr.keys()))
    n_files_test = len(list(data_danra_test_zarr.keys()))

    # Get LSM and topography data
    PATH_LSM_FULL = args.path_data + 'data_lsm/truth_fullDomain/lsm_full.npz'
    PATH_TOPO_FULL = args.path_data + 'data_topo/truth_fullDomain/topo_full.npz'

    data_lsm_full = np.flipud(np.load(PATH_LSM_FULL)['data'])
    data_topo_full = np.flipud(np.load(PATH_TOPO_FULL)['data'])

    # If scaling chosen, scale topography to interval [0,1] - based on the min/max in cutout domains of northern NE/DL/PL and DK
    if scaling:
        topo_min, topo_max = -12, 330
        norm_min, norm_max = 0, 1
        
        OldRange = (topo_max - topo_min)
        NewRange = (norm_max - norm_min)

        # Generating the new data based on the given intervals
        data_topo_full = (((data_topo_full - topo_min) * NewRange) / OldRange) + norm_min

    # Define data hyperparameters
    input_channels = args.in_channels
    output_channels = args.out_channels




    n_samples_train = n_files_train
    n_samples_valid = n_files_valid
    n_samples_test = n_files_test
    
    cache_size = args.cache_size
    if cache_size == 0:
        cache_size_train = n_samples_train//2
        cache_size_valid = n_samples_valid//2
        cache_size_test = n_samples_test//2
    else:
        cache_size_train = cache_size
        cache_size_valid = cache_size
        cache_size_test = cache_size
    print(f'\n\n\nNumber of training samples: {n_samples_train}')
    print(f'Number of validation samples: {n_samples_valid}')
    print(f'Number of test samples: {n_samples_test}\n')
    print(f'Total number of samples: {n_samples_train + n_samples_valid + n_samples_test}\n\n\n')

    print(f'\n\n\nCache size for training: {cache_size_train}')
    print(f'Cache size for validation: {cache_size_valid}')
    print(f'Cache size for test: {cache_size_test}\n')
    print(f'Total cache size: {cache_size_train + cache_size_valid + cache_size_test}\n\n\n')


    image_size = args.HR_SHAPE
    image_dim = image_size[0]
    
    n_seasons = args.season_shape[0]
    if n_seasons != 0:
        condition_on_seasons = True
    else:
        condition_on_seasons = False

    loss_type = args.loss_type
    if loss_type == 'sdfweighted':
        sdf_weighted_loss = True
    else:
        sdf_weighted_loss = False

    config_name = args.config_name
    save_str = config_name + var + '__' + str(image_dim) + 'x' + str(image_dim) + '__' + loss_type + '__' + str(n_seasons) + '_seasons'

    create_figs = args.create_figs
    
    # Set path to save figures
    PATH_SAMPLES = PATH_SAVE + f'/Samples' + '__' + config_name
    PATH_LOSSES = PATH_SAVE + '/losses'
    
    if not os.path.exists(PATH_SAMPLES):
        os.makedirs(PATH_SAMPLES)
    if not os.path.exists(PATH_LOSSES):
        os.makedirs(PATH_LOSSES)

    NAME_SAMPLES = 'Generated_samples' + '__' + save_str + '__' + 'epoch' + '_'
    NAME_FINAL_SAMPLES = f'Final_generated_sample' + '__' + save_str
    NAME_LOSSES = f'Training_losses' + '__' + save_str


    # Define the path to the pretrained model 
    PATH_CHECKPOINT = args.path_checkpoint
    try:
        os.makedirs(PATH_CHECKPOINT)
        print('\n\n\nCreating directory for saving checkpoints...')
        print(f'Directory created at {PATH_CHECKPOINT}')
    except FileExistsError:
        print('\n\n\nDirectory for saving checkpoints already exists...')
        print(f'Directory located at {PATH_CHECKPOINT}')


    NAME_CHECKPOINT = save_str + '.pth.tar'

    checkpoint_dir = PATH_CHECKPOINT
    checkpoint_name = NAME_CHECKPOINT
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f'\nCheckpoint path: {checkpoint_path}')


    # Define model hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    first_fmap_channels = args.first_fmap_channels
    last_fmap_channels = args.last_fmap_channels
    time_embedding = args.time_embedding_size
    
    learning_rate = args.lr
    min_lr = args.min_lr
    weight_decay = args.weight_decay

    # Define diffusion hyperparameters
    n_timesteps = args.n_timesteps
    beta_min = args.beta_range[0]
    beta_max = args.beta_range[1]
    beta_scheduler = args.beta_scheduler
    noise_variance = args.noise_variance

    # Define if samples should be moved around in the cutout domains
    CUTOUTS = args.CUTOUTS
    CUTOUT_DOMAINS = args.CUTOUT_DOMAINS

    # Define the loss function
    if loss_type == 'simple':
        lossfunc = SimpleLoss()
        use_sdf_weighted_loss = False
    elif loss_type == 'hybrid':
        lossfunc = HybridLoss(alpha=0.5, T=n_timesteps)#nn.MSELoss()#SimpleLoss()#
        use_sdf_weighted_loss = False
    elif loss_type == 'sdfweighted':
        lossfunc = SDFWeightedMSELoss(max_land_weight=1.0, min_sea_weight=0.0)
        use_sdf_weighted_loss = True
        # NEED TO ACCOUNT FOR POSSIBILITY OF MULTIPLE DOMAINS


    if args.LR_VARS is not None:
        condition_on_img = True

    
    # Define training dataset, with cutouts enabled and data from zarr files
    train_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_train_w_cutouts_zarr, 
                                            data_size = image_size, 
                                            n_samples = n_samples_train, 
                                            cache_size = cache_size_train, 
                                            variable=var,
                                            scale=scaling, 
                                            conditional_seasons=condition_on_seasons,
                                            conditional_images=condition_on_img,
                                            cond_dir_zarr=data_dir_era5_train_zarr, 
                                            n_classes=n_seasons, 
                                            cutouts=CUTOUTS, 
                                            cutout_domains=CUTOUT_DOMAINS,
                                            lsm_full_domain=data_lsm_full,
                                            topo_full_domain=data_topo_full,
                                            sdf_weighted_loss = use_sdf_weighted_loss
                                            )
    # Define validation dataset, with cutouts enabled and data from zarr files
    valid_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_valid_w_cutouts_zarr, 
                                            data_size = image_size, 
                                            n_samples = n_samples_valid, 
                                            cache_size = cache_size_valid, 
                                            variable=var,
                                            scale=scaling,
                                            conditional_seasons=condition_on_seasons, 
                                            conditional_images=condition_on_img,
                                            cond_dir_zarr=data_dir_era5_valid_zarr,
                                            n_classes=n_seasons, 
                                            cutouts=CUTOUTS, 
                                            cutout_domains=CUTOUT_DOMAINS,
                                            lsm_full_domain=data_lsm_full,
                                            topo_full_domain=data_topo_full,
                                            sdf_weighted_loss = use_sdf_weighted_loss
                                            )
    # Define test dataset, with cutouts enabled and data from zarr files
    gen_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_test_w_cutouts_zarr,
                                            data_size = image_size,
                                            n_samples = n_samples_test,
                                            cache_size = cache_size_test,
                                            variable=var,
                                            scale=scaling,
                                            shuffle=True,
                                            conditional_seasons=condition_on_seasons, 
                                            conditional_images=condition_on_img,    
                                            cond_dir_zarr=data_dir_era5_test_zarr,
                                            n_classes=n_seasons,
                                            cutouts=CUTOUTS,
                                            cutout_domains=CUTOUT_DOMAINS,
                                            lsm_full_domain=data_lsm_full,
                                            topo_full_domain=data_topo_full,
                                            sdf_weighted_loss = use_sdf_weighted_loss
                                            )


    # Define the torch dataloaders for train and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)#, num_workers=args.num_workers)

    n_gen_samples = args.n_gen_samples
    gen_dataloader = DataLoader(gen_dataset, batch_size=n_gen_samples, shuffle=False)#, num_workers=args.num_workers)

    # Examine first batch from test dataloader

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
    encoder = Encoder(input_channels, 
                        time_embedding,
                        cond_on_img=condition_on_img, 
                        cond_img_dim=(1, args.LR_SHAPE[0], args.LR_SHAPE[1]), 
                        block_layers=[2, 2, 2, 2], 
                        num_classes=n_seasons,
                        n_heads=args.num_heads
                        )
    decoder = Decoder(last_fmap_channels, 
                        output_channels, 
                        time_embedding, 
                        first_fmap_channels,
                        n_heads=args.num_heads
                        )
    # Define the model from modules_DANRA_downscaling.py
    model = DiffusionNet(encoder, decoder)


    # Define the diffusion utils from diffusion_DANRA_downscaling.py
    diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, device, beta_scheduler)        

    # Define the optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)

    # Define the training pipeline
    pipeline = TrainingPipeline_general(model,
                                        lossfunc,
                                        optimizer,
                                        diffusion_utils,
                                        device,
                                        weight_init=True,
                                        sdf_weighted_loss=sdf_weighted_loss
                                        )

    # Define the learning rate scheduler
    if args.lr_scheduler is not None:
        lr_scheduler_params = args.lr_scheduler_params
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer,
                                                                'min',
                                                                factor=lr_scheduler_params['factor'],
                                                                patience=lr_scheduler_params['patience'],
                                                                threshold=lr_scheduler_params['threshold'],
                                                                verbose=True,
                                                                min_lr=min_lr
                                                                )

    # Check if the path to pretrained model exists
    if os.path.isfile(checkpoint_path):
        # If the path exists, load the pretrained weights
        print('\n\n\nLoading pretrained weights...')
        print(checkpoint_path)
        checkpoint_state = torch.load(checkpoint_path, map_location=device)['network_params']
        pipeline.model.load_state_dict(checkpoint_state)


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
        
        # Print epoch number
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
        print(f'\n\nTraining Loss: {train_loss:.6f}\n\n')
        print(f'Validation Loss: {valid_loss:.6f}\n\n')

        # If valid loss is better than best loss, save model. With early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            pipeline.save_model(checkpoint_dir, checkpoint_name)
            print(f'Model saved at epoch {epoch+1} with training loss {train_loss:.6f}')
            print(f'Validation loss: {valid_loss:.6f}')
            print(f'Saved to {checkpoint_dir} with name {checkpoint_name}')
            
            # Early stopping
            PATIENCE = args.early_stopping_params['patience']
        else:
            PATIENCE -= 1
            if PATIENCE == 0:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Step the learning rate scheduler
        lr_scheduler.step(train_loss)

        # If n_gen_samples is 0, continue to next epoch
        if n_gen_samples == 0:
            continue
        # Else, generate samples
        else:
            # Create samples and plot if valid loss is better than best loss, and if epoch > 5
            if (valid_loss < best_loss and epoch > 0) or epoch == 0:
                # Generate 1 sample for n different test images
                if epoch == 0:
                    print('First epoch, generating samples...')
                else:
                    print('Valid loss is better than best loss, generating samples...')

                for idx, samples in tqdm.tqdm(enumerate(gen_dataloader), total=len(gen_dataloader)):
                    # Extract samples from dictionary
                    test_images, test_seasons, test_cond, test_lsm, test_sdf, test_topo, _ = extract_samples(samples, device=device)
                    data_plot = [test_images, test_cond, test_lsm, test_sdf, test_topo]
                    data_names = ['Truth', 'Condition', 'LSM', 'SDF', 'Topography']
                    # Filter out None samples
                    data_plot = [sample for sample in data_plot if sample is not None]
                    data_names = [name for name, sample in zip(data_names, data_plot) if sample is not None]

                    # Count length of data_plot
                    n_axs = len(data_plot)

                    # Generate image from model
                    generated_image = diffusion_utils.sample(n=n_gen_samples,
                                                            model=pipeline.model,
                                                            channels_hr=1,
                                                            y=test_seasons,
                                                            cond_img=test_cond,
                                                            lsm_cond=test_lsm,
                                                            topo_cond=test_topo
                                                            )
                    generated_image = generated_image.detach().cpu()

                    data_plot.append(generated_image)
                    data_names.append('Generated')

                    fig, axs = plt.subplots(n_axs+1, n_gen_samples, figsize=(14,9)) # Plotting truth, condition, generated, lsm and topo for n different test images
                    
                    # Make the first row the generated images
                    for i in range(n_gen_samples):
                        img = data_plot[-1][i].squeeze()
                        image = axs[0, i].imshow(img, cmap=cmap_name)
                        axs[0, i].set_title(f'{data_names[-1]}')
                        axs[0, i].axis('off')
                        axs[0, i].set_ylim([0, img.shape[0]])
                        fig.colorbar(image, ax=axs[0, i], fraction=0.046, pad=0.04, orientation='vertical')


                    # Loop through the generated samples (and corresponding truth, condition, lsm and topo) and plot
                    for i in range(n_gen_samples):
                        for j in range(n_axs):
                            img = data_plot[j][i].squeeze()
                            if data_names[j] == 'Truth' or data_names[j] == 'Condition':
                                cmap_name_use = cmap_name
                            else:
                                cmap_name_use = 'viridis'
                            image = axs[j+1, i].imshow(img, cmap=cmap_name_use)
                            axs[j+1, i].set_title(f'{data_names[j]}')
                            axs[j+1, i].axis('off')
                            axs[j+1, i].set_ylim([0, img.shape[0]])
                            fig.colorbar(image, ax=axs[j+1, i], fraction=0.046, pad=0.04)

                    fig.tight_layout()
                    #plt.show()
                    
                    # Save figure
                    if epoch == (epochs - 1):
                        fig.savefig(PATH_SAMPLES + '/' + NAME_FINAL_SAMPLES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
                    else:
                        fig.savefig(PATH_SAMPLES + '/' + NAME_SAMPLES + str(epoch+1) + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
                    
                    break


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
            

        






# if __name__ == '__main__':
#     freeze_support()

#     print('\n\n')
#     print('#'*50)
#     print('Running ddpm')
#     print('#'*50)
#     print('\n\n')


#     # General settings for use

#     # Define DANRA data information 
#     # Set variable for use
#     var = 'temp'#'prcp'#

#     # Set scaling to true or false
#     scaling = True

#     # Set some color map settings
#     if var == 'temp':
#         cmap_name = 'plasma'
#         cmap_label = 'Temperature [°C]'
#     elif var == 'prcp':
#         cmap_name = 'inferno'
#         cmap_label = 'Precipitation [mm]'


#     # Set size of DANRA images
#     n_danra_size = 64 #32 #256#
#     # Set DANRA size string for use in path
#     danra_size_str = '589x789'#str(n_danra_size) + 'x' + str(n_danra_size)

#     # Path to DANRA data (zarr files), full danra, to enable cutouts
#     data_dir_danra_train_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_train.zarr'
#     data_dir_danra_valid_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_valid.zarr'
#     data_dir_danra_test_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_test.zarr'

#     # Path to ERA5 data, 589x789 (same size as DANRA)
#     data_dir_era5_train_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_train.zarr'
#     data_dir_era5_valid_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_valid.zarr'
#     data_dir_era5_test_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_test.zarr'
    
#     # Make zarr groups
#     data_danra_train_zarr = zarr.open_group(data_dir_danra_train_w_cutouts_zarr, mode='r')
#     data_danra_valid_zarr = zarr.open_group(data_dir_danra_valid_w_cutouts_zarr, mode='r')
#     data_danra_test_zarr = zarr.open_group(data_dir_danra_test_w_cutouts_zarr, mode='r')

#     data_era5_train_zarr = zarr.open_group(data_dir_era5_train_zarr, mode='r')
#     data_era5_valid_zarr = zarr.open_group(data_dir_era5_valid_zarr, mode='r')
#     data_era5_test_zarr = zarr.open_group(data_dir_era5_test_zarr, mode='r')


#     n_files_train = len(list(data_danra_train_zarr.keys()))
#     n_files_valid = len(list(data_danra_valid_zarr.keys()))
#     n_files_test = len(list(data_danra_test_zarr.keys()))

#     # Get LSM and topography data
#     PATH_LSM_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
#     PATH_TOPO_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

#     data_lsm_full = np.flipud(np.load(PATH_LSM_FULL)['data'])
#     data_topo_full = np.flipud(np.load(PATH_TOPO_FULL)['data'])

#     # If scaling chosen, scale topography to interval [0,1] - based on the min/max in cutout domains of northern NE/DL/PL and DK
#     if scaling:
#         topo_min, topo_max = -12, 330
#         norm_min, norm_max = 0, 1
        
#         OldRange = (topo_max - topo_min)
#         NewRange = (norm_max - norm_min)

#         # Generating the new data based on the given intervals
#         data_topo_full = (((data_topo_full - topo_min) * NewRange) / OldRange) + norm_min

#     # Define data hyperparameters
#     input_channels = 1
#     output_channels = 1

#     n_samples_train = n_files_train
#     cache_size_train = n_files_train//2

#     n_samples_valid = n_files_valid
#     cache_size_valid = n_files_valid

#     n_samples_test = n_files_test
#     cache_size_test = n_files_test

#     print(f'\n\n\nNumber of training samples: {n_samples_train}')
#     print(f'Number of validation samples: {n_samples_valid}')
#     print(f'Number of test samples: {n_samples_test}\n')
#     print(f'Total number of samples: {n_samples_train + n_samples_valid + n_samples_test}\n\n\n')

#     print(f'\n\n\nCache size for training: {cache_size_train}')
#     print(f'Cache size for validation: {cache_size_valid}')
#     print(f'Cache size for test: {cache_size_test}\n')
#     print(f'Total cache size: {cache_size_train + cache_size_valid + cache_size_test}\n\n\n')


#     image_dim = n_danra_size#64#32#64#
#     image_size = (image_dim,image_dim)
#     n_seasons = 4#12#366#
#     loss_type = 'sdfweighted'#'simple'#'hybrid'#

#     # Define strings for use in path
#     scale_str = '_ZscoreScaled' if scaling else ''
#     im_dim_str = str(image_dim) + 'x' + str(image_dim)
#     cond_str = 'ERA5_cond_lsm_topo_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs' + scale_str + '_test'
#     var_str = var
#     model_str = 'DDPM_conditional_ERA5'
#     # Set path to save figures
#     SAVE_FIGS = False
#     PATH_SAVE = '/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_clean/samples'
#     PATH_SAMPLES = PATH_SAVE + f'/Samples' + '__' + var_str + '__' + im_dim_str + '__' + cond_str 
#     PATH_LOSSES = '/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_clean/losses'
#     if not os.path.exists(PATH_SAMPLES):
#         os.makedirs(PATH_SAMPLES)

#     NAME_SAMPLES = 'Generated_samples' + '__' + var_str + '__' + im_dim_str + '__' + cond_str + '__' + 'epoch' + '_'
#     NAME_FINAL_SAMPLES = f'Final_generated_sample' + '__' + var_str + '__' + im_dim_str + '__' + cond_str 
#     NAME_LOSSES = f'Training_losses' + '__' + var_str + '__' + im_dim_str + '__' + cond_str


#     # Define the path to the pretrained model 
#     PATH_CHECKPOINT = '/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_clean/model_checkpoints/'
#     try:
#         os.makedirs(PATH_CHECKPOINT)
#         print('\n\n\nCreating directory for saving checkpoints...')
#         print(f'Directory created at {PATH_CHECKPOINT}')
#     except FileExistsError:
#         print('\n\n\nDirectory for saving checkpoints already exists...')
#         print(f'Directory located at {PATH_CHECKPOINT}')


#     NAME_CHECKPOINT = model_str + '__' + var_str + '__' + im_dim_str + '__' + cond_str + '.pth.tar'

#     checkpoint_dir = PATH_CHECKPOINT
#     checkpoint_name = NAME_CHECKPOINT
#     checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
#     print(f'\nCheckpoint path: {checkpoint_path}')

#     PLOT_EPOCH_SAMPLES = True


#     # Define model hyperparameters
#     epochs = 50
#     batch_size = 32
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     first_fmap_channels = 64#n_danra_size #
#     last_fmap_channels = 512 #2048
#     time_embedding = 256
#     learning_rate = 3e-4 #1e-2
#     min_lr = 1e-6
#     weight_decay = 0.0

#     # Define diffusion hyperparameters
#     n_timesteps = 1000#800
#     beta_min = 1e-4
#     beta_max = 0.02
#     beta_scheduler = 'cosine'#'linear'

#     # Define if samples should be moved around in the cutout domains
#     CUTOUTS = True
#     CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]



#     # Define training dataset, with cutouts enabled and data from zarr files
#     train_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_train_w_cutouts_zarr, 
#                                             data_size = image_size, 
#                                             n_samples = n_samples_train, 
#                                             cache_size = cache_size_train, 
#                                             variable=var,
#                                             scale=scaling, 
#                                             conditional_seasons=True,
#                                             conditional_images=True,
#                                             cond_dir_zarr=data_dir_era5_train_zarr, 
#                                             n_classes=n_seasons, 
#                                             cutouts=CUTOUTS, 
#                                             cutout_domains=CUTOUT_DOMAINS,
#                                             lsm_full_domain=data_lsm_full,
#                                             topo_full_domain=data_topo_full,
#                                             sdf_weighted_loss = True
#                                             )
#     # Define validation dataset, with cutouts enabled and data from zarr files
#     valid_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_valid_w_cutouts_zarr, 
#                                             data_size = image_size, 
#                                             n_samples = n_samples_valid, 
#                                             cache_size = cache_size_valid, 
#                                             variable=var,
#                                             scale=scaling,
#                                             conditional_seasons=True, 
#                                             conditional_images=True,
#                                             cond_dir_zarr=data_dir_era5_valid_zarr,
#                                             n_classes=n_seasons, 
#                                             cutouts=CUTOUTS, 
#                                             cutout_domains=CUTOUT_DOMAINS,
#                                             lsm_full_domain=data_lsm_full,
#                                             topo_full_domain=data_topo_full,
#                                             sdf_weighted_loss = True
#                                             )
#     # Define test dataset, with cutouts enabled and data from zarr files
#     test_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_test_w_cutouts_zarr,
#                                             data_size = image_size,
#                                             n_samples = n_samples_test,
#                                             cache_size = cache_size_test,
#                                             variable=var,
#                                             scale=scaling,
#                                             shuffle=True,
#                                             conditional_seasons=True, 
#                                             conditional_images=True,    
#                                             cond_dir_zarr=data_dir_era5_test_zarr,
#                                             n_classes=n_seasons,
#                                             cutouts=CUTOUTS,
#                                             cutout_domains=CUTOUT_DOMAINS,
#                                             lsm_full_domain=data_lsm_full,
#                                             topo_full_domain=data_topo_full,
#                                             sdf_weighted_loss = True
#                                             )


#     # Define the torch dataloaders for train and validation
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
#     valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

#     n_test_samples = 4
#     test_dataloader = DataLoader(test_dataset, batch_size=n_test_samples, shuffle=False, num_workers=3)

#     # Examine first batch from test dataloader

#     # Define the seed for reproducibility, and set seed for torch, numpy and random
#     seed = 42
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     np.random.seed(seed)

#     # Set torch to deterministic mode, meaning that the same input will always produce the same output
#     torch.backends.cudnn.deterministic = False
#     # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
#     torch.backends.cudnn.benchmark = True

#     # Define the encoder and decoder from modules_DANRA_downscaling.py
#     encoder = Encoder(input_channels, 
#                         time_embedding,
#                         cond_on_img=True, 
#                         cond_img_dim=(1, image_size[0], image_size[1]), 
#                         block_layers=[2, 2, 2, 2], 
#                         num_classes=n_seasons,
#                         n_heads=8#image_dim//2
#                         )
#     decoder = Decoder(last_fmap_channels, 
#                         output_channels, 
#                         time_embedding, 
#                         first_fmap_channels,
#                         n_heads=8#image_dim//2
#                         )
#     # Define the model from modules_DANRA_downscaling.py
#     model = DiffusionNet(encoder, decoder)


#     # Define the diffusion utils from diffusion_DANRA_downscaling.py
#     diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, device, beta_scheduler)
#     # Define the loss function

#     if loss_type == 'simple':
#         lossfunc = SimpleLoss()
#     elif loss_type == 'hybrid':
#         lossfunc = HybridLoss(alpha=0.5, T=n_timesteps)#nn.MSELoss()#SimpleLoss()#
#     elif loss_type == 'sdfweighted':
#         lossfunc = SDFWeightedMSELoss(max_land_weight=1.0, min_sea_weight=0.0)
#         # NEED TO ACCOUNT FOR POSSIBILITY OF MULTIPLE DOMAINS
        

#     # Define the optimizer
#     optimizer = torch.optim.AdamW(model.parameters(),
#                                   lr=learning_rate,
#                                   weight_decay=weight_decay)

#     # Define the training pipeline
#     pipeline = TrainingPipeline_general(model,
#                                         lossfunc,
#                                         optimizer,
#                                         diffusion_utils,
#                                         device,
#                                         weight_init=True,
#                                         sdf_weighted_loss=True
#                                         )

#     # Define the learning rate scheduler
#     lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer, 'min', factor=0.5, patience=5, verbose=True, min_lr=min_lr)
#     #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(pipeline.optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=len(train_dataloader), pct_start=0.3, anneal_strategy='cos', final_div_factor=300)
#     #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#     #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
#     #lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, epochs=N, steps_per_epoch=len(data_loader), pct_start=0.3, anneal_strategy='cos', final_div_factor=300)
#     #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)    

#     # Check if the path to pretrained model exists
#     if os.path.isfile(checkpoint_path):
#         # If the path exists, load the pretrained weights
#         print('\n\n\nLoading pretrained weights...')
#         print(checkpoint_path)
#         checkpoint_state = torch.load(checkpoint_path, map_location=device)['network_params']
#         pipeline.model.load_state_dict(checkpoint_state)


#     # Check if device is cuda, if so print information and empty cache
#     if torch.cuda.is_available():
#         print(f'Model is training on {torch.cuda.get_device_name()}\n\n')
#         print(f'Model is using {torch.cuda.memory_allocated()} bytes of memory\n\n')
#         torch.cuda.empty_cache()

#     # Set empty list for train losses
#     train_losses = []
#     valid_losses = []

#     # Set best loss to infinity
#     best_loss = np.inf

#     print(f'\n\n\nStarting training...\n\n')
#     # Loop over epochs
#     for epoch in range(epochs):
        
#         # Print epoch number
#         print(f'\n\nEpoch {epoch+1} of {epochs}:\n\n')
#         if epoch == 0:
#             PLOT_FIRST_IMG = True
#         else:
#             PLOT_FIRST_IMG = False

#         # Call the train function from pipeline
#         train_loss = pipeline.train(train_dataloader, verbose=False, PLOT_FIRST=PLOT_FIRST_IMG, SAVE_PATH=PATH_SAMPLES, SAVE_NAME='upsampled_images_example.png')
#         # Append train loss to list
#         train_losses.append(train_loss)
        
#         # Call the evaluate function from pipeline
#         valid_loss = pipeline.validate(valid_dataloader, verbose=False)
#         # Append valid loss to list
#         valid_losses.append(valid_loss)

#         # # Print train and valid loss
#         print(f'\n\nTraining Loss: {train_loss:.6f}\n\n')
#         print(f'Validation Loss: {valid_loss:.6f}\n\n')

#         # If valid loss is better than best loss, save model. With early stopping
#         if valid_loss < best_loss:
#             best_loss = valid_loss
#             pipeline.save_model(checkpoint_dir, checkpoint_name)
#             print(f'Model saved at epoch {epoch+1} with training loss {train_loss:.6f}')
#             print(f'Validation loss: {valid_loss:.6f}')
#             print(f'Saved to {checkpoint_dir} with name {checkpoint_name}')
            
#             # Early stopping
#             PATIENCE = 5
#         else:
#             PATIENCE -= 1
#             if PATIENCE == 0:
#                 print(f'Early stopping at epoch {epoch+1}')
#                 break
        
#         # Create samples and plot if valid loss is better than best loss, and if epoch > 5
#         if (valid_loss < best_loss and epoch > 0) or epoch == 0:
#             # Generate 1 sample for n different test images
#             if epoch == 0:
#                 print('First epoch, generating samples...')
#             else:
#                 print('Valid loss is better than best loss, generating samples...')

#             # Set number of samples to generate (equal to batch size of test dataloader)
#             n = n_test_samples

#             for idx, samples in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
#                 # Extract samples from dictionary
#                 test_images, test_seasons, test_cond, test_lsm, test_sdf, test_topo, _ = extract_samples(samples, device=device)
#                 data_plot = [test_images, test_cond, test_lsm, test_sdf, test_topo]
#                 data_names = ['Truth', 'Condition', 'LSM', 'SDF', 'Topography']
#                 # Filter out None samples
#                 data_plot = [sample for sample in data_plot if sample is not None]
#                 data_names = [name for name, sample in zip(data_names, data_plot) if sample is not None]

#                 # Count length of data_plot
#                 n_axs = len(data_plot)

#                 # Generate image from model
#                 generated_image = diffusion_utils.sample(n=n_test_samples,
#                                                         model=pipeline.model,
#                                                         channels_hr=1,
#                                                         y=test_seasons,
#                                                         cond_img=test_cond,
#                                                         lsm_cond=test_lsm,
#                                                         topo_cond=test_topo
#                                                         )
#                 generated_image = generated_image.detach().cpu()

#                 data_plot.append(generated_image)
#                 data_names.append('Generated')

#                 fig, axs = plt.subplots(n_axs+1, n, figsize=(14,9)) # Plotting truth, condition, generated, lsm and topo for n different test images
                
#                 # Make the first row the generated images
#                 for i in range(n):
#                     img = data_plot[-1][i].squeeze()
#                     image = axs[0, i].imshow(img, cmap=cmap_name)
#                     axs[0, i].set_title(f'{data_names[-1]}')
#                     axs[0, i].axis('off')
#                     axs[0, i].set_ylim([0, img.shape[0]])
#                     fig.colorbar(image, ax=axs[0, i], fraction=0.046, pad=0.04, orientation='vertical')


#                 # Loop through the generated samples (and corresponding truth, condition, lsm and topo) and plot
#                 for i in range(n_test_samples):
#                     for j in range(n_axs):
#                         img = data_plot[j][i].squeeze()
#                         if data_names[j] == 'Truth' or data_names[j] == 'Condition':
#                             cmap_name_use = cmap_name
#                         else:
#                             cmap_name_use = 'viridis'
#                         image = axs[j+1, i].imshow(img, cmap=cmap_name_use)
#                         axs[j+1, i].set_title(f'{data_names[j]}')
#                         axs[j+1, i].axis('off')
#                         axs[j+1, i].set_ylim([0, img.shape[0]])
#                         fig.colorbar(image, ax=axs[j+1, i], fraction=0.046, pad=0.04)

#                 fig.tight_layout()
#                 #plt.show()
                
#                 # Save figure
#                 if epoch == (epochs - 1):
#                     fig.savefig(PATH_SAMPLES + '/' + NAME_FINAL_SAMPLES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
#                 else:
#                     fig.savefig(PATH_SAMPLES + '/' + NAME_SAMPLES + str(epoch+1) + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
                
#                 break

                


#             fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#             ax.plot(train_losses, label='Train')
#             ax.plot(valid_losses, label='Validation')
#             ax.set_xlabel('Epoch')
#             ax.set_ylabel('Loss')
#             ax.set_title('Loss')
#             ax.legend(loc='upper right')
#             fig.savefig(PATH_SAMPLES + '/' + NAME_LOSSES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

#             with open(PATH_LOSSES + '/' + NAME_LOSSES + '_train', 'wb') as fp:
#                 pickle.dump(train_losses, fp)
#             with open(PATH_LOSSES + '/' + NAME_LOSSES + '_valid', 'wb') as fp:
#                 pickle.dump(valid_losses, fp)
            

#         # Step the learning rate scheduler
#         lr_scheduler.step(train_loss)





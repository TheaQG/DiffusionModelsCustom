'''
    Script to perform training of DDPM on DANRA data, with conditional training on season labels.
    With validation set and saving of model checkpoints.

    Based on https://github.com/dome272/Diffusion-Models-pytorch 
    and
    https://arxiv.org/abs/2006.11239 (DDPM)
'''

import os, torch
import pickle
import zarr
import numpy as np

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


# Import objects from other files in this repository
from data_DANRA_conditional import DANRA_Dataset_cutouts_ERA5_Zarr, preprocess_lsm_topography
from modules_DANRA_conditional import *
from diffusion_DANRA_conditional import DiffusionUtils
from training_DANRA_conditional import *

# Run 'export MKL_SERVICE_FORCE_INTEL=1' in bash





if __name__ == '__main__':
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
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
    n_danra_size = 32 #128# 256#
    # Set DANRA size string for use in path
    danra_size_str = '589x789'#str(n_danra_size) + 'x' + str(n_danra_size)
    # Set paths to data
    data_dir_danra_full = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str
    data_dir_danra_train = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str + '_train'
    data_dir_danra_valid = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str + '_valid'

    # Path to training (full danra, to enable cutouts)
    data_dir_danra_train_w_cutouts = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/' + var + '_' + danra_size_str + '_train'
    # Path to validation (full danra, to enable cutouts)
    data_dir_danra_valid_w_cutouts = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/' + var + '_' + danra_size_str + '_valid'
    # Path to test (full danra, to enable cutouts)
    data_dir_danra_test_w_cutouts = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/' + var + '_' + danra_size_str + '_test'

    # Path to train ERA5 data, 589x789 (same size as DANRA)
    data_dir_era5_train = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/' + var + '_589x789_train'
    # Path to validation ERA5 data, 589x789 (same size as DANRA)
    data_dir_era5_valid = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/' + var + '_589x789_valid'
    # Path to test ERA5 data, 589x789 (same size as DANRA)
    data_dir_era5_test = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/' + var + '_589x789_test'


    # Path to zarr files
    # Path to training (full danra, to enable cutouts)
    data_dir_danra_train_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_train.zarr'
    # Path to validation (full danra, to enable cutouts)
    data_dir_danra_valid_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_valid.zarr'
    # Path to test (full danra, to enable cutouts)
    data_dir_danra_test_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_test.zarr'

    # Path to train ERA5 data, 589x789 (same size as DANRA)
    data_dir_era5_train_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_train.zarr'
    # Path to validation ERA5 data, 589x789 (same size as DANRA)
    data_dir_era5_valid_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_valid.zarr'
    # Path to test ERA5 data, 589x789 (same size as DANRA)
    data_dir_era5_test_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_test.zarr'
    # Make zarr groups
    data_danra_train_zarr = zarr.open_group(data_dir_danra_train_w_cutouts_zarr, mode='r')
    data_danra_valid_zarr = zarr.open_group(data_dir_danra_valid_w_cutouts_zarr, mode='r')
    data_danra_test_zarr = zarr.open_group(data_dir_danra_test_w_cutouts_zarr, mode='r')

    data_era5_train_zarr = zarr.open_group(data_dir_era5_train_zarr, mode='r')
    data_era5_valid_zarr = zarr.open_group(data_dir_era5_valid_zarr, mode='r')
    data_era5_test_zarr = zarr.open_group(data_dir_era5_test_zarr, mode='r')

    n_files_train = len(list(data_danra_train_zarr.keys()))
    n_files_valid = len(list(data_danra_valid_zarr.keys()))
    n_files_test = len(list(data_danra_test_zarr.keys()))

    # n_files_train = 0
    # for root, _, files in os.walk(data_dir_danra_train_w_cutouts):
    #     for name in files:
    #         if name.endswith('.npz') or name.endswith('.nc'):
    #             n_files_train += 1

    # n_files_valid = 0
    # for root, _, files in os.walk(data_dir_danra_valid_w_cutouts):
    #     for name in files:
    #         if name.endswith('.npz') or name.endswith('.nc'):
    #             n_files_valid += 1

    # n_files_test = 0
    # for root, _, files in os.walk(data_dir_danra_test_w_cutouts):
    #     for name in files:
    #         if name.endswith('.npz') or name.endswith('.nc'):
    #             n_files_test += 1

    # Define data hyperparameters
    input_channels = 1
    output_channels = 1


    n_samples_train = n_files_train
    cache_size_train = n_files_train//2

    n_samples_valid = n_files_valid
    cache_size_valid = n_files_valid

    n_samples_test = n_files_test
    cache_size_test = n_files_test

    print(f'\n\n\nNumber of training samples: {n_samples_train}')
    print(f'Number of validation samples: {n_samples_valid}')
    print(f'Number of test samples: {n_samples_test}\n')
    print(f'Total number of samples: {n_samples_train + n_samples_valid + n_samples_test}\n\n\n')

    print(f'\n\n\nCache size for training: {cache_size_train}')
    print(f'Cache size for validation: {cache_size_valid}')
    print(f'Cache size for test: {cache_size_test}\n')
    print(f'Total cache size: {cache_size_train + cache_size_valid + cache_size_test}\n\n\n')

    image_dim = n_danra_size#64#32#64#
    image_size = (image_dim,image_dim)
    n_seasons = 4#12#366#
    loss_type = 'sdfweighted'#'simple'#'hybrid'#
    p_train = 0.8 # Train split
    # p_valid = 0.2 # Validation split 
    # p_test = 0.0 # Test split

    # Define strings for use in path
    im_dim_str = str(image_dim) + 'x' + str(image_dim)
    cond_str = 'BOTTLENECKTEST__ERA5_cond_lsm_topo_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs'
    var_str = var
    model_str = 'DDPM_conditional_ERA5'
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
    epochs = 1
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

    lsm_tensor, topo_tensor = preprocess_lsm_topography(PATH_LSM, PATH_TOPO, image_size, scale=False, flip=True)#, scale=True)
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
            im_lsm = ax.imshow(lsm_tensor.squeeze(), cmap='viridis')
            ax.set_title(f'Land-sea mask, upscaled to {image_size[0]}x{image_size[1]}')
            ax.set_ylim(ax.get_ylim()[::-1])
            fig.colorbar(im_lsm, ax=ax, fraction=0.046, pad=0.04)
        elif topo_tensor is not None:
            im_topo = ax.imshow(topo_tensor.squeeze(), cmap='viridis')
            ax.set_title(f'Topography, upscaled to {image_size[0]}x{image_size[1]}')
            ax.set_ylim(ax.get_ylim()[::-1])
            fig.colorbar(im_topo, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(PATH_SAMPLES + f'/ddpm_conditional__{var}_lsm_topo.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

    elif n_plots == 2:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        im_lsm = axs[0].imshow(lsm_tensor.squeeze(), cmap='viridis')
        axs[0].set_title(f'Land-sea mask, upscaled to {image_size[0]}x{image_size[1]}')
        axs[0].set_ylim(axs[0].get_ylim()[::-1])
        fig.colorbar(im_lsm, ax=axs[0], fraction=0.046, pad=0.04)

        im_topo = axs[1].imshow(topo_tensor.squeeze(), cmap='viridis')
        axs[1].set_title(f'Topography, upscaled to {image_size[0]}x{image_size[1]}')
        axs[1].set_ylim(axs[1].get_ylim()[::-1])
        fig.colorbar(im_topo, ax=axs[1], fraction=0.046, pad=0.04)
        
        print(f'\n\n\nSaving lsm and topography figure...')
        print('\n\n')
        fig.savefig(PATH_SAMPLES + f'/ddpm_conditional__{var}_lsm_topo.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

    elif n_plots == 0:
        print(f'\n\n\nNo lsm or topography tensor found...\n\n')
        print(f'Continuing without lsm and topography tensors...\n\n')

    CUTOUTS = True
    CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]

    PATH_LSM_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
    PATH_TOPO_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

    data_lsm_full = np.flipud(np.load(PATH_LSM_FULL)['data'])
    data_topo_full = np.flipud(np.load(PATH_TOPO_FULL)['data'])



    # Define training dataset, with cutouts enabled and data from zarr files
    train_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_train_w_cutouts_zarr, 
                                            data_size = image_size, 
                                            n_samples = n_samples_train, 
                                            cache_size = cache_size_train, 
                                            variable=var,
                                            scale=False, 
                                            shuffle=False, 
                                            conditional=True,
                                            cond_dir_zarr=data_dir_era5_train_zarr, 
                                            n_classes=n_seasons, 
                                            cutouts=CUTOUTS, 
                                            cutout_domains=CUTOUT_DOMAINS,
                                            lsm_full_domain=data_lsm_full,
                                            topo_full_domain=data_topo_full,
                                            sdf_weighted_loss = True
                                            )
    # Define validation dataset, with cutouts enabled and data from zarr files
    valid_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_valid_w_cutouts_zarr, 
                                            data_size = image_size, 
                                            n_samples = n_samples_valid, 
                                            cache_size = cache_size_valid, 
                                            variable=var,
                                            scale=False, 
                                            shuffle=False, 
                                            conditional=True, 
                                            cond_dir_zarr=data_dir_era5_valid_zarr,
                                            n_classes=n_seasons, 
                                            cutouts=CUTOUTS, 
                                            cutout_domains=CUTOUT_DOMAINS,
                                            lsm_full_domain=data_lsm_full,
                                            topo_full_domain=data_topo_full,
                                            sdf_weighted_loss = True
                                            )
    # Define test dataset, with cutouts enabled and data from zarr files
    test_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_test_w_cutouts_zarr,
                                            data_size = image_size,
                                            n_samples = n_samples_test,
                                            cache_size = cache_size_test,
                                            variable=var,
                                            scale=False,
                                            shuffle=True,
                                            conditional=True,
                                            cond_dir_zarr=data_dir_era5_test_zarr,
                                            n_classes=n_seasons,
                                            cutouts=CUTOUTS,
                                            cutout_domains=CUTOUT_DOMAINS,
                                            lsm_full_domain=data_lsm_full,
                                            topo_full_domain=data_topo_full,
                                            sdf_weighted_loss = True
                                            )
    # Define the dataset from data_DANRA_downscaling.py
    # train_dataset = DANRA_Dataset(data_dir_danra_train,
    #                               image_size,
    #                               n_samples_train,
    #                               cache_size_train,
    #                               scale=False,
    #                               shuffle=False,
    #                               conditional=True,
    #                               n_classes=n_seasons)
    # valid_dataset = DANRA_Dataset(data_dir_danra_valid,
    #                               image_size,
    #                               n_samples_valid,
    #                               cache_size_valid,
    #                               scale=False,
    #                               shuffle=False,
    #                               conditional=True,
    #                               n_classes=n_seasons)

    # Define training dataset, with cutouts enabled
    # train_dataset = DANRA_Dataset_cutouts_ERA5(data_dir = data_dir_danra_train_w_cutouts, 
    #                                         data_size = image_size, 
    #                                         n_samples = n_samples_train, 
    #                                         cache_size = cache_size_train, 
    #                                         variable=var,
    #                                         scale=False, 
    #                                         shuffle=False, 
    #                                         conditional=True,
    #                                         data_dir_cond = data_dir_era5_train, 
    #                                         n_classes=n_seasons, 
    #                                         cutouts=CUTOUTS, 
    #                                         cutout_domains=CUTOUT_DOMAINS,
    #                                         lsm_full_domain=data_lsm_full,
    #                                         topo_full_domain=data_topo_full,
    #                                         sdf_weighted_loss = True
    #                                         )
    # # Define validation dataset, with cutouts enabled
    # valid_dataset = DANRA_Dataset_cutouts_ERA5(data_dir = data_dir_danra_valid_w_cutouts, 
    #                                         data_size = image_size, 
    #                                         n_samples = n_samples_valid, 
    #                                         cache_size = cache_size_valid, 
    #                                         variable=var,
    #                                         scale=False, 
    #                                         shuffle=False, 
    #                                         conditional=True, 
    #                                         data_dir_cond = data_dir_era5_valid,
    #                                         n_classes=n_seasons, 
    #                                         cutouts=CUTOUTS, 
    #                                         cutout_domains=CUTOUT_DOMAINS,
    #                                         lsm_full_domain=data_lsm_full,
    #                                         topo_full_domain=data_topo_full,
    #                                         sdf_weighted_loss = True
    #                                         )
    # # Define test dataset, with cutouts enabled
    # test_dataset = DANRA_Dataset_cutouts_ERA5(data_dir = data_dir_danra_test_w_cutouts,
    #                                         data_size = image_size,
    #                                         n_samples = n_samples_test,
    #                                         cache_size = cache_size_test,
    #                                         variable=var,
    #                                         scale=False,
    #                                         shuffle=True,
    #                                         conditional=True,
    #                                         data_dir_cond = data_dir_era5_test,
    #                                         n_classes=n_seasons,
    #                                         cutouts=CUTOUTS,
    #                                         cutout_domains=CUTOUT_DOMAINS,
    #                                         lsm_full_domain=data_lsm_full,
    #                                         topo_full_domain=data_topo_full,
    #                                         sdf_weighted_loss = True
    #                                         )





    ####################
    # FOR RANDOM SPLIT #
    ####################
    # # Split the dataset into train and validation
    # train_size = int(p_train * len(train_dataset))
    # valid_size = len(train_dataset) - train_size

    # train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])



    # Define the torch dataloaders for train and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    n_test_samples = 4
    test_dataloader = DataLoader(test_dataset, batch_size=n_test_samples, shuffle=False, num_workers=1)

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
                        lsm_tensor=lsm_tensor, 
                        topo_tensor=topo_tensor, 
                        cond_on_img=True, 
                        cond_img_dim=(1, image_size[0], image_size[1]), 
                        block_layers=[2, 2, 2, 2], 
                        num_classes=n_seasons)
    decoder = Decoder(last_fmap_channels, 
                        output_channels, 
                        time_embedding, 
                        first_fmap_channels)
    # Define the model from modules_DANRA_downscaling.py
    model = DiffusionNet(encoder, decoder)
    # Define the diffusion utils from diffusion_DANRA_downscaling.py
    diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, device, beta_scheduler)
    # Define the loss function

    if loss_type == 'simple':
        lossfunc = SimpleLoss()
    elif loss_type == 'hybrid':
        lossfunc = HybridLoss(alpha=0.5, T=n_timesteps)#nn.MSELoss()#SimpleLoss()#
    elif loss_type == 'sdfweighted':
        lossfunc = SDFWeightedMSELoss(max_land_weight=1.0, min_sea_weight=0.0)
        # NEED TO ACCOUNT FOR POSSIBILITY OF MULTIPLE DOMAINS
        

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay)
    # Define the training pipeline from training_DANRA_downscaling.py
    if loss_type == 'simple':
        pipeline = TrainingPipeline_ERA5_Condition(model,
                                                    lossfunc,
                                                    optimizer,
                                                    diffusion_utils,
                                                    device,
                                                    weight_init=True
                                                    )
    elif loss_type == 'hybrid':
        pipeline = TrainingPipeline_Hybrid(model,
                                            lossfunc,
                                            optimizer,
                                            diffusion_utils,
                                            device,
                                            weight_init=True
                                            )
    elif loss_type == 'sdfweighted':
        pipeline = TrainingPipeline_ERA5_Condition(model,
                                                    lossfunc,
                                                    optimizer,
                                                    diffusion_utils,
                                                    device,
                                                    weight_init=True,
                                                    sdf_weighted_loss=True
                                                    )

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
        cond_img = torch.randn(n_samples, 1, image_size[0], image_size[1]).to(device)
        lsm_test = torch.ones(n_samples, 1, image_size[0], image_size[1]).to(device)
        topo_test = torch.ones(n_samples, 1, image_size[0], image_size[1]).to(device)
        #idx_test = np.random.randint(0, n_seasons)
        #y_test[idx_test] = 1
        y_test = y_test.type(torch.int64).to(device)

        print(f'Input shape of test: {test_input.shape}')
        print(f'Shape of time embedding: {t_test.shape}')
        print(f'Shape of season embedding: {y_test.shape}')
        test = pipeline.model(test_input, t_test, y_test, cond_img=cond_img, lsm_cond=lsm_test, topo_cond=topo_test)

        
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
        print(f'\n\nTraining Loss: {train_loss:.6f}\n\n')
        print(f'Validation Loss: {valid_loss:.6f}\n\n')

        # If valid loss is better than best loss, save model
        if valid_loss < best_loss:
            best_loss = valid_loss
            pipeline.save_model(checkpoint_dir, checkpoint_name)
            print(f'Model saved at epoch {epoch+1}')# with training loss {train_loss:.6f}')
            print(f'Validation loss: {valid_loss:.6f}')
            print(f'Saved to {checkpoint_dir} with name {checkpoint_name}')


        # If epoch is multiple of 10 or last epoch, generate samples from test set
        if (epoch == 0 and PLOT_EPOCH_SAMPLES) or (((epoch + 1) % 10) == 0 and PLOT_EPOCH_SAMPLES) or (epoch == (epochs - 1) and PLOT_EPOCH_SAMPLES):
            # Generate 1 sample for n different test images
            print('Generating samples...')
            
            # Set number of samples to generate (equal to batch size of test dataloader)
            n = n_test_samples
            fig, axs = plt.subplots(5, n, figsize=(14,9)) # Plotting truth, condition, generated, lsm and topo for n different test images

            for idx, samples in enumerate(test_dataloader):
                if loss_type == 'sdfweighted':
                    (test_img, test_season, test_cond), test_lsm, test_topo, test_sdf, _ = samples
                else:
                    (test_img, test_season, test_cond), test_lsm, test_topo, _ = samples

                print(f'\n\n\nShape of test image: {test_img.shape}')
                print(f'Shape of test season: {test_season.shape}')
                print(f'Shape of test condition: {test_cond.shape}')
                print(f'Shape of test lsm: {test_lsm.shape}')
                if loss_type == 'sdfweighted':
                    print(f'Shape of test topo: {test_topo.shape}')
                    print(f'Shape of test sdf: {test_sdf.shape}\n\n')
                else:
                    print(f'Shape of test topo: {test_topo.shape}\n\n')

                # Generate random fields of same shape as test image and send to device
                x = torch.randn(n, input_channels, *image_size).to(device)
                # Send all other parts of sample to device
                test_season = test_season.to(device)
                test_cond = test_cond.to(torch.float).to(device)
                test_lsm = test_lsm.to(device)
                test_topo = test_topo.to(device)
                test_sdf = test_sdf.to(device)


                # Print the shapes and types of the different tensors
                print(f'\n\n\nShape of test truth image: {test_img.shape}')
                print(f'Type: {test_img.dtype}')
                print(f'Shape of noise: {x.shape}')
                print(f'Type: {x.dtype}')
                print(f'Shape of test season: {test_season.shape}')
                print(f'Type: {test_season.dtype}')
                print(f'Shape of test condition: {test_cond.shape}')
                print(f'Type: {test_cond.dtype}')
                print(f'Shape of test lsm: {test_lsm.shape}')
                print(f'Type: {test_lsm.dtype}')
                print(f'Shape of test topo: {test_topo.shape}')
                print(f'Type: {test_topo.dtype}\n\n')


                # Generate image from model
                generated_image = diffusion_utils.sample(x, pipeline.model, test_season, cond_img=test_cond, lsm_cond=test_lsm, topo_cond=test_topo)
                generated_image = generated_image.detach().cpu()

                # Loop through the generated samples (and corresponding truth, condition, lsm and topo) and plot
                for i in range(n_test_samples):
                    img_truth = test_img[i].squeeze()
                    img_cond = test_cond[i].squeeze()
                    img_gen = generated_image[i].squeeze()
                    img_lsm = test_lsm[i].squeeze()
                    img_topo = test_topo[i].squeeze()

                    image_truth = axs[0, i].imshow(img_truth, cmap='viridis')
                    axs[0, i].set_title(f'Truth')
                    axs[0, i].axis('off')
                    axs[0, i].set_ylim([0, img_truth.shape[0]])
                    fig.colorbar(image_truth, ax=axs[0, i], fraction=0.046, pad=0.04)
                    
                    image_cond = axs[1, i].imshow(img_cond, cmap='viridis')
                    axs[1, i].set_title(f'Condition')
                    axs[1, i].axis('off')
                    axs[1, i].set_ylim([0, img_cond.shape[0]])
                    fig.colorbar(image_cond, ax=axs[1, i], fraction=0.046, pad=0.04)

                    image_gen = axs[2, i].imshow(img_gen, cmap='viridis')
                    axs[2, i].set_title(f'Generated')
                    axs[2, i].axis('off')
                    axs[2, i].set_ylim([0, img_gen.shape[0]])
                    fig.colorbar(image_gen, ax=axs[2, i], fraction=0.046, pad=0.04)

                    image_lsm = axs[3, i].imshow(img_lsm, cmap='viridis')
                    axs[3, i].set_title(f'LSM')
                    axs[3, i].axis('off')
                    axs[3, i].set_ylim([0, img_lsm.shape[0]])
                    fig.colorbar(image_lsm, ax=axs[3, i], fraction=0.046, pad=0.04)

                    image_topo = axs[4, i].imshow(img_topo, cmap='viridis')
                    axs[4, i].set_title(f'Topography')
                    axs[4, i].axis('off')
                    axs[4, i].set_ylim([0, img_topo.shape[0]])
                    fig.colorbar(image_topo, ax=axs[4, i], fraction=0.046, pad=0.04)

                fig.tight_layout()
                #plt.show()
                
                # Save figure
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
            

        # Step the learning rate scheduler
        lr_scheduler.step(train_loss)

    # Load best model state
    best_model_path = checkpoint_path#os.path.join('../../ModelCheckpoints/DDPM_DANRA', 'DDPM.pth.tar')
    best_model_state = torch.load(best_model_path)['network_params']

    # Load best model state into model
    pipeline.model.load_state_dict(best_model_state)

    print('Generating samples...')

    # Set number of final samples to generate
    n = 8

    # Create a figure for plotting
    fig, axs = plt.subplots(5, n, figsize=(18,8)) # Plotting truth, condition, generated, lsm and topo for n different test images

    # Make a dataloader with batch size equal to n
    final_dataloader = DataLoader(test_dataset, batch_size=n, shuffle=False, num_workers=1)

    # Generate samples from final dataloader
    for idx, samples in enumerate(final_dataloader):
        print(samples)
        if loss_type == 'sdfweighted':
            (test_img, test_season, test_cond), test_lsm, test_topo, test_sdf, _ = samples
        else:
            (test_img, test_season, test_cond), test_lsm, test_topo, _ = samples

        # Generate random fields of same shape as test image and send to device
        x = torch.randn(n, input_channels, *image_size).to(device)
        # Send all other parts of sample to device
        test_season = test_season.to(device)
        test_cond = test_cond.to(torch.float).to(device)
        test_lsm = test_lsm.to(device)
        test_topo = test_topo.to(device)
        if loss_type == 'sdfweighted':
            test_sdf = test_sdf.to(device)


        # Print the shapes and types of the different tensors
        print(f'\n\n\nShape of test truth image: {test_img.shape}')
        print(f'Type: {test_img.dtype}')
        print(f'Shape of noise: {x.shape}')
        print(f'Type: {x.dtype}')
        print(f'Shape of test season: {test_season.shape}')
        print(f'Type: {test_season.dtype}')
        print(f'Shape of test condition: {test_cond.shape}')
        print(f'Type: {test_cond.dtype}')
        print(f'Shape of test lsm: {test_lsm.shape}')
        print(f'Type: {test_lsm.dtype}')
        print(f'Shape of test topo: {test_topo.shape}')
        print(f'Type: {test_topo.dtype}\n\n')
        if loss_type == 'sdfweighted':
            print(f'Shape of eval sdf: {test_sdf.shape}')
            print(f'Type: {test_sdf.dtype}\n\n')


        # Generate image from model
        generated_image = diffusion_utils.sample(x,
                                                 pipeline.model,
                                                 test_season,
                                                 cond_img=test_cond,
                                                 lsm_cond=test_lsm,
                                                 topo_cond=test_topo
                                                 )
        generated_image = generated_image.detach().cpu()

        # Loop through the generated samples (and corresponding truth, condition, lsm and topo) and plot
        for i in range(n_test_samples):
            img_truth = test_img[i].squeeze()
            img_cond = test_cond[i].squeeze()
            img_gen = generated_image[i].squeeze()
            img_lsm = test_lsm[i].squeeze()
            img_topo = test_topo[i].squeeze()

            image_truth = axs[0, i].imshow(img_truth, cmap='viridis')
            axs[0, i].set_title(f'Truth')
            axs[0, i].axis('off')
            axs[0, i].set_ylim([0, img_truth.shape[0]])
            fig.colorbar(image_truth, ax=axs[0, i], fraction=0.046, pad=0.04)
            
            image_cond = axs[1, i].imshow(img_cond, cmap='viridis')
            axs[1, i].set_title(f'Condition')
            axs[1, i].axis('off')
            axs[1, i].set_ylim([0, img_cond.shape[0]])
            fig.colorbar(image_cond, ax=axs[1, i], fraction=0.046, pad=0.04)

            image_gen = axs[2, i].imshow(img_gen, cmap='viridis')
            axs[2, i].set_title(f'Generated')
            axs[2, i].axis('off')
            axs[2, i].set_ylim([0, img_gen.shape[0]])
            fig.colorbar(image_gen, ax=axs[2, i], fraction=0.046, pad=0.04)

            image_lsm = axs[3, i].imshow(img_lsm, cmap='viridis')
            axs[3, i].set_title(f'LSM')
            axs[3, i].axis('off')
            axs[3, i].set_ylim([0, img_lsm.shape[0]])
            fig.colorbar(image_lsm, ax=axs[3, i], fraction=0.046, pad=0.04)

            image_topo = axs[4, i].imshow(img_topo, cmap='viridis')
            axs[4, i].set_title(f'Topography')
            axs[4, i].axis('off')
            axs[4, i].set_ylim([0, img_topo.shape[0]])
            fig.colorbar(image_topo, ax=axs[4, i], fraction=0.046, pad=0.04)

        fig.tight_layout()
        plt.show()

        # Save figure
        fig.savefig(PATH_SAMPLES + '/' + NAME_FINAL_SAMPLES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)







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



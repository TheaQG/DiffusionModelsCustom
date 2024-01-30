'''
    Script to run evaluations of trained ddpm models on test DANRA dataset.
    Default evalutes on evaluation set of size equal to two years of data (730 samples), 2001-2002.
    The default size is 64x64.

'''

import torch
import zarr
import os
import pysal

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import summary

from modules_DANRA_conditional import *
from diffusion_DANRA_conditional import DiffusionUtils
from training_DANRA_conditional import *
from data_DANRA_conditional import preprocess_lsm_topography, DANRA_Dataset_cutouts_ERA5_Zarr
from daily_files_to_zarr import convert_npz_to_zarr, convert_nc_to_zarr

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # Define DANRA data information 
    # Set variable for use
    var = 'temp'#'prcp'# 

    # Set DANRA size string for use in path
    danra_size_str = '589x789'

    # Set the size of the images 
    danra_size = 64

    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)


    # First choose the data to evaluate on (n random samples from the years 2001-2005)
    n_samples_gen = 100
    year_start = 2001
    year_end = 2005

    # Define data hyperparameters
    input_channels = 1
    output_channels = 1
    n_samples = n_samples_gen
    cache_size = n_samples_gen
    image_size = (danra_size, danra_size)
    n_seasons = None#4
    CUTOUTS = True
    CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]

    PATH_LSM_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
    PATH_TOPO_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

    data_lsm_full = np.flipud(np.load(PATH_LSM_FULL)['data'])
    data_topo_full = np.flipud(np.load(PATH_TOPO_FULL)['data'])

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
    loss_type = 'simple'#'sdfweighted'#'simple'#


    # Define diffusion hyperparameters
    n_timesteps = 800
    beta_min = 1e-4
    beta_max = 0.02
    beta_scheduler = 'linear'



    eval_npz_dir_era5 = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/temp_589x789_eval'
    eval_nc_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789_full/temp_589x789_eval'

    # Path tp zarr dirs
    data_dir_danra_eval_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_eval.zarr'
    data_dir_era5_eval_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_' + danra_size_str + '/zarr_files/' + var + '_' + danra_size_str + '_eval.zarr'

    # Convert .nc files in DANRA eval dir to zarr files
    convert_nc_to_zarr(eval_nc_dir_danra, data_dir_danra_eval_w_cutouts_zarr)
    # Convert .npz files in ERA5 eval dir to zarr files
    convert_npz_to_zarr(eval_npz_dir_era5, data_dir_era5_eval_zarr)

    # Create zarr groups
    data_danra_eval_zarr = zarr.open_group(data_dir_danra_eval_w_cutouts_zarr, mode='r')
    data_era5_eval_zarr = zarr.open_group(data_dir_era5_eval_zarr, mode='r')




    # Create evaluation dataset
    eval_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_eval_w_cutouts_zarr,
                                                    data_size = image_size,
                                                    n_samples = n_samples,
                                                    cache_size = cache_size,
                                                    variable=var,
                                                    scale=False,
                                                    shuffle=True,
                                                    conditional_seasons=False,
                                                    conditional_images=False,
                                                    cond_dir_zarr=None,
                                                    n_classes=None,
                                                    cutouts=CUTOUTS,
                                                    cutout_domains=CUTOUT_DOMAINS,
                                                    sdf_weighted_loss = False
                                                    )

    # Preprocess lsm and topography (normalize and reshape, resizes to image_size)
    PATH_LSM = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_DK/lsm_dk.npz'
    PATH_TOPO = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_DK/topo_dk.npz'


    lsm_tensor, topo_tensor = preprocess_lsm_topography(PATH_LSM, PATH_TOPO, image_size, scale=False, flip=True)#, scale=True)
    lsm_weight = 1
    lsm_tensor = lsm_weight * lsm_tensor#None#
    topo_weight = 1
    topo_tensor = topo_weight * topo_tensor#None#

    encoder = Encoder(input_channels, 
                            time_embedding, 
                            lsm_tensor=None, 
                            topo_tensor=None, 
                            cond_on_img=False, 
                            cond_img_dim=None,#(1, image_size[0], image_size[1]), 
                            block_layers=[2, 2, 2, 2], 
                            num_classes=None#n_seasons
                            )
    decoder = Decoder(last_fmap_channels, 
                            output_channels, 
                            time_embedding, 
                            first_fmap_channels)

    # Define model
    model = DiffusionNet(encoder, decoder)

    # Define diffusion 
    diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, device, beta_scheduler)


    
    # Define the loss function
    if loss_type == 'simple':
        lossfunc = SimpleLoss()
    elif loss_type == 'hybrid':
        lossfunc = HybridLoss(alpha=0.5, T=n_timesteps)#nn.MSELoss()#SimpleLoss()#
    elif loss_type == 'sdfweighted':
        lossfunc = SDFWeightedMSELoss(max_land_weight=1.0, min_sea_weight=0.0)
        # NEED TO ACCOUNT FOR POSSIBILITY OF MULTIPLE DOMAINS

    # Define optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Define training pipeline
    if loss_type == 'simple':
        pipeline = TrainingPipeline_general(model,
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
        pipeline = TrainingPipeline_general(model,
                                            lossfunc,
                                            optimizer,
                                            diffusion_utils,
                                            device,
                                            weight_init=True,
                                            sdf_weighted_loss=True
                                            )



    # Define path to model checkpoint
    image_dim = danra_size
    im_dim_str = str(image_dim) + 'x' + str(image_dim)
    cond_str = 'unconditional_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs'
    var_str = var
    model_str = 'DDPM_unconditional'
    checkpoint_dir = "/Users/au728490/Documents/PhD_AU/Python_Scripts/ModelCheckpoints/LUMI_trained_models"
    NAME_CHECKPOINT = model_str + '__' + var_str + '__' + im_dim_str + '__' + cond_str + '.pth.tar'
    
    checkpoint_path = os.path.join(checkpoint_dir, NAME_CHECKPOINT)


    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path)
    best_model_path = os.path.join(checkpoint_dir, NAME_CHECKPOINT)
    best_model_state = torch.load(best_model_path)['network_params']

    # Load best state model into model
    pipeline.model.load_state_dict(best_model_state)
    pipeline.model.eval()

    # Print the summary of the model
    print('\n\nModel summary:')
    #x, t, y, cond_img, lsm_cond, topo_cond
    # input_size_summary = (input_channels, image_size[0], image_size[1]), (1,), (1,), (1, image_size[0], image_size[1]), (1, image_size[0], image_size[1]), (1, image_size[0], image_size[1])
    # summary(model, input_size=[input_size_summary], batch_size=batch_size, device=device)


    print("Generating samples...")


    # Set number of samples to generate
    n = n_samples_gen

    # Make a dataloader with batch size equal to n
    eval_dataloader = DataLoader(eval_dataset, batch_size=n, shuffle=False, num_workers=1)

    # Generate samples from evaluation dataloader
    
    for idx, samples in enumerate(eval_dataloader):
        # Generate random fields of same shape as test image and send to device
        x = torch.randn(n, input_channels, *image_size).to(device)
        print(f'Shape of noise: {x.shape}')
        print(f'Type: {x.dtype}')
        print('\n\n')                

        # Get parts of samples and send to device
        if 'img' in samples.keys():
            eval_img = samples['img'].to(device)
            print(f'Shape of test truth image: {eval_img.shape}')
            print(f'Type: {eval_img.dtype}')
        else:
            # Print error, you need to have images
            print('Error: you need to have images in your dataset')

        if 'classifier' in samples.keys():
            eval_seasons = samples['classifier'].to(device)
            print(f'Shape of test season: {eval_seasons.shape}')
            print(f'Type: {eval_seasons.dtype}')
        else:
            eval_seasons = None
            print('No season str condition')
        
        if 'img_cond' in samples.keys():
            eval_cond = samples['img_cond']
            eval_cond = eval_cond.to(torch.float).to(device)
            print(f'Shape of test condition: {eval_cond.shape}')
            print(f'Type: {eval_cond.dtype}')
        else:
            eval_cond = None
            print('No conditional image')
        
        if 'lsm' in samples.keys():
            eval_lsm = samples['lsm'].to(device)
            eval_lsm = eval_lsm.to(device)
            print(f'Shape of test lsm: {eval_lsm.shape}')
            print(f'Type: {eval_lsm.dtype}')
        else:
            eval_lsm = None
            print('No lsm')

        if 'sdf' in samples.keys():
            eval_sdf = samples['sdf'].to(device)
            print(f'Shape of test sdf: {eval_sdf.shape}')
            print(f'Type: {eval_sdf.dtype}')
        else:
            eval_sdf = None
            print('No sdf')
        
        if 'topo' in samples.keys():
            eval_topo = samples['topo'].to(device)
            eval_topo = eval_topo.to(device)
            print(f'Shape of test topo: {eval_topo.shape}')
            print(f'Type: {eval_topo.dtype}')
        else:
            eval_topo = None
            print('No topography')

        if 'points' in samples.keys():
            points = samples['points']
            print(f'Shape of test points: {len(points)}')
            print(f'Type: {points[0].dtype}')
        else:
            points = None
            print('No points')

        # Generate image from model
        generated_images = diffusion_utils.sample(x,
                                                pipeline.model,
                                                eval_seasons,
                                                cond_img=eval_cond,
                                                lsm_cond=eval_lsm,
                                                topo_cond=eval_topo
                                                )
        generated_image = generated_images.detach().cpu()



        # Stop after first iteration, all samples are generated
        break

    # Save the generated and corresponding eval images
    SAVE_PATH = '/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_DANRA_conditional/final_generated_samples/'
    SAVE_NAME =  model_str + '__' + var_str + '__' + im_dim_str + '__' + cond_str + '__' + str(n_samples_gen) + '_samples.npz'
    print(f'\n\nSaving generated and eval images to {SAVE_PATH}...')
    print(f'Saving as {SAVE_PATH + "gen_samples__" + SAVE_NAME}')
    np.savez_compressed(SAVE_PATH + 'gen_samples__' + SAVE_NAME, generated_images)
    np.savez_compressed(SAVE_PATH + 'eval_samples__' + SAVE_NAME, eval_img)
    np.savez_compressed(SAVE_PATH + 'lsm_samples__' + SAVE_NAME, eval_lsm)
    np.savez_compressed(SAVE_PATH + 'cond_samples__' + SAVE_NAME, eval_cond)
    np.savez_compressed(SAVE_PATH + 'season_samples__' + SAVE_NAME, eval_seasons)
    np.savez_compressed(SAVE_PATH + 'point_samples__' + SAVE_NAME, points)




'''
    Script to run evaluations of trained ddpm models on test DANRA dataset.
    Default evalutes on evaluation set of size equal to two years of data (730 samples), 2001-2002.
    The default size is 64x64.

    Need to develop evaluation methods:
    - Full test set generation + evaluation
    - Single sample generation + evaluation
    - Repeated single sample generation + evaluation
    


'''

import torch
import zarr
import os
import pysal

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import summary

from ..src.unet import *
from ..src.diffusion_modules import DiffusionUtils
from ..src.plot_utils import *
from ..src.data_modules import *
from ..src.utils import *
from ..src.training import *



if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()


    # Set variable for use
    var = 'temp'#'prcp'# 
    scaling = True

    # Set DANRA size string for use in path
    danra_size_str = '589x789'

    # Set the size of the images 
    danra_size = 64

    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)


    # First choose the data to evaluate on (n random samples from the years 2001-2005)
    n_samples_gen = 50
    year_start = 2001
    year_end = 2005

    # Select folder with .nc/.npz files for generation
    gen_dir_era5 = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_' + danra_size_str + '/' + var + '_' + danra_size_str
    gen_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/' + var + '_' + danra_size_str

    # List files in the directories in chronological order
    gen_files_era5 = os.listdir(gen_dir_era5)
    gen_files_danra = os.listdir(gen_dir_danra)

    # ERA5 startrs with 2000, DANRA with 2001
    # DANRA files are named as't2m_ave_YYYYMMDD.nc', ERA5 as 'temp_589x789_YYYYMMDD.npz'
    # Select only files from the years 2001-2005 and avoid .DS_Store file
    gen_files_era5 = [file for file in gen_files_era5 if (file != '.DS_Store') and (int(file[-12:-8]) >= year_start) and (int(file[-12:-8]) <= year_end)]
    gen_files_danra = [file for file in gen_files_danra if (file != '.DS_Store') and (int(file[-11:-7]) >= year_start) and (int(file[-11:-7]) <= year_end)]
    print(f'\n\nNumber of files in ERA5 generation dataset: {len(gen_files_era5)}')
    print(f'Number of files in DANRA generation dataset: {len(gen_files_danra)}')

    # Find the files that are not in both datasets (based on date)
    gen_files_era5_dates = [int(file[-12:-4]) for file in gen_files_era5]
    gen_files_danra_dates = [int(file[-11:-3]) for file in gen_files_danra]

    gen_files_era5_not_in_danra = [file for file in gen_files_era5_dates if file not in gen_files_danra_dates]
    gen_files_danra_not_in_era5 = [file for file in gen_files_danra_dates if file not in gen_files_era5_dates]
    print(f'\n\nFiles in ERA5 not in DANRA: {gen_files_era5_not_in_danra}')
    print(f'Files in DANRA not in ERA5: {gen_files_danra_not_in_era5}')

    print('\n\nRemoving files not in both datasets...')
    # Remove the files that are not in both datasets
    gen_files_era5 = sorted([file for file in gen_files_era5 if int(file[-12:-4]) not in gen_files_era5_not_in_danra])
    gen_files_danra = sorted([file for file in gen_files_danra if int(file[-11:-3]) not in gen_files_danra_not_in_era5])

    # Make lists with the updated dates
    gen_files_era5_dates = [int(file[-12:-4]) for file in gen_files_era5]
    gen_files_danra_dates = [int(file[-11:-3]) for file in gen_files_danra]

    # Sort the dates and check that the dates lists are equal
    gen_files_era5_dates.sort()
    gen_files_danra_dates.sort()
    assert gen_files_era5_dates == gen_files_danra_dates, "The dates in the two datasets are not equal!"

    print('\n\nThe dates in the two datasets are now the same!')
    print(f'Number of files in ERA5 generation dataset: {len(gen_files_era5)}')
    print(f'Number of files in DANRA generation dataset: {len(gen_files_danra)}')


    # Select n random dates (the same for both datasets) to generate samples from
    print(f'\n\nSelecting {n_samples_gen} random dates to generate samples from...')
    
    # Check that there are enough files to generate n_samples_gen samples
    if n_samples_gen > len(gen_files_era5_dates):
        # If not, generate as many samples as there are files
        print(f'Not enough files to generate {n_samples_gen} samples, generating {len(gen_files_era5_dates)} samples instead!')
        n_samples_gen = len(gen_files_era5_dates)

    # Select n random dates from the dates in the two datasets
    gen_dates = np.random.choice(gen_files_era5_dates, size=n_samples_gen, replace=False)
    # Sort the dates
    gen_dates.sort()

    print('\n\nRandom dates selected:')
    print(gen_dates)
    # Find the indices of the selected dates in the two datasets
    gen_files_era5_idx = [gen_files_era5_dates.index(date) for date in gen_dates]
    gen_files_danra_idx = [gen_files_danra_dates.index(date) for date in gen_dates]
    # Check that the indices are the same
    print('\n\nIndices of selected dates:')
    print(gen_files_era5_idx==gen_files_danra_idx)


    # Select the files corresponding to the selected dates
    gen_files_era5 = [gen_files_era5[idx] for idx in gen_files_era5_idx]
    gen_files_danra = [gen_files_danra[idx] for idx in gen_files_danra_idx]

    print(f'\n\nSelected files, n = {len(gen_files_era5)}:')
    # Print the selected files
    for i in range(len(gen_files_era5)):
        print(f'ERA5 file: {gen_files_era5[i]}')
        print(f'DANRA file: {gen_files_danra[i]}\n')

    # Check the seasonality of the selected dates 
    print('\n\nChecking seasonality of selected dates...')
    winter = ['12', '01', '02']
    spring = ['03', '04', '05']
    summer = ['06', '07', '08']
    autumn = ['09', '10', '11']

    # Sum the number of dates in each season
    winter_count = sum([1 for date in gen_dates if str(date)[-4:-2] in winter])
    spring_count = sum([1 for date in gen_dates if str(date)[-4:-2] in spring])
    summer_count = sum([1 for date in gen_dates if str(date)[-4:-2] in summer])
    autumn_count = sum([1 for date in gen_dates if str(date)[-4:-2] in autumn])

    print(f'\n\nWinter count: {winter_count}')
    print(f'Spring count: {spring_count}')
    print(f'Summer count: {summer_count}')
    print(f'Autumn count: {autumn_count}')



    # Now copy selected files to eval directories
    print('\n\nCopying selected files to eval directories...')

    eval_npz_dir_era5 = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/temp_589x789_eval'
    eval_nc_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789_full/temp_589x789_eval'


    # Check if eval directories exist, if not create them. If not empty, empty them
    if not os.path.exists(eval_npz_dir_era5):
        os.mkdir(eval_npz_dir_era5)
    else:
        for file in os.listdir(eval_npz_dir_era5):
            os.remove(os.path.join(eval_npz_dir_era5, file))

    if not os.path.exists(eval_nc_dir_danra):
        os.mkdir(eval_nc_dir_danra)

    else:
        for file in os.listdir(eval_nc_dir_danra):
            os.remove(os.path.join(eval_nc_dir_danra, file))

    # Copy the selected files to the eval directories
    for i in range(len(gen_files_era5)):
        os.system(f'cp {os.path.join(gen_dir_era5, gen_files_era5[i])} {os.path.join(eval_npz_dir_era5, gen_files_era5[i])}')
        os.system(f'cp {os.path.join(gen_dir_danra, gen_files_danra[i])} {os.path.join(eval_nc_dir_danra, gen_files_danra[i])}')

    print(f'\n\nN files in ERA5 eval directory: {len(os.listdir(eval_npz_dir_era5))}')
    print(f'N files in DANRA eval directory: {len(os.listdir(eval_nc_dir_danra))}')

    # Create zarr files from the copied files
    print('\n\nCreating zarr files from copied files...')

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



    # Define data hyperparameters
    input_channels = 1
    output_channels = 1
    n_samples = n_samples_gen
    cache_size = n_samples_gen
    image_size = (danra_size, danra_size)
    image_dim = danra_size
    n_seasons = 4
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
    loss_type = 'sdfweighted'#'simple'#'simple'#


    # Define diffusion hyperparameters
    n_timesteps = 800
    beta_min = 1e-4
    beta_max = 0.02
    beta_scheduler = 'linear'

    # Create evaluation dataset
    if loss_type == 'sdfweighted':
        eval_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_eval_w_cutouts_zarr,
                                                       data_size = image_size,
                                                       n_samples = n_samples,
                                                       cache_size = cache_size,
                                                       variable=var,
                                                       shuffle=True,
                                                       cutouts=CUTOUTS,
                                                       cutout_domains=CUTOUT_DOMAINS,
                                                       lsm_full_domain=data_lsm_full,
                                                       topo_full_domain=data_topo_full,
                                                       sdf_weighted_loss = True,
                                                       scale=scaling,
                                                       conditional_seasons=True,
                                                       conditional_images=True,
                                                       cond_dir_zarr=data_dir_era5_eval_zarr,
                                                       n_classes=n_seasons
                                                       )
    else:
        eval_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_eval_w_cutouts_zarr,
                                                       data_size = image_size,
                                                       n_samples = n_samples,
                                                       cache_size = cache_size,
                                                       variable=var,
                                                       shuffle=True,
                                                       cutouts=CUTOUTS,
                                                       cutout_domains=CUTOUT_DOMAINS,
                                                       lsm_full_domain=data_lsm_full,
                                                       topo_full_domain=data_topo_full,
                                                       sdf_weighted_loss = False,
                                                       scale=scaling,
                                                       conditional_seasons=True,
                                                       conditional_images=True,
                                                       cond_dir_zarr=data_dir_era5_eval_zarr,
                                                       n_classes=n_seasons
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
                            cond_on_img=True, 
                            cond_img_dim=(1, image_size[0], image_size[1]), 
                            block_layers=[2, 2, 2, 2], 
                            num_classes=n_seasons,
                            n_heads=8#image_dim//2
                            )
    decoder = Decoder(last_fmap_channels, 
                            output_channels, 
                            time_embedding, 
                            first_fmap_channels,
                            n_heads=8#image_dim//2
                            )

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
    pipeline = TrainingPipeline_general(model,
                                        lossfunc,
                                        optimizer,
                                        diffusion_utils,
                                        device,
                                        weight_init=True,
                                        sdf_weighted_loss=True
                                        )

    # Define path to model checkpoint
    scale_str = '_ZscoreScaled' if scaling else ''
    image_dim = danra_size
    im_dim_str = str(image_dim) + 'x' + str(image_dim)
    cond_str = 'ERA5_cond_lsm_topo_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs' + scale_str + '_test'
    var_str = var
    model_str = 'DDPM_conditional_ERA5'
    checkpoint_dir = '/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_clean/model_checkpoints'
    NAME_CHECKPOINT = model_str + '__' + var_str + '__' + im_dim_str + '__' + cond_str + '.pth.tar'
    
    checkpoint_path = os.path.join(checkpoint_dir, NAME_CHECKPOINT)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f'\n\nCheckpoint {checkpoint_path} does not exist, exiting...')
        exit(1)


    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path)
    best_model_path = os.path.join(checkpoint_dir, NAME_CHECKPOINT)
    best_model_state = torch.load(best_model_path)['network_params']

    # Load best state model into model
    pipeline.model.load_state_dict(best_model_state)
    pipeline.model.eval()

    # Print the summary of the model
    # print('\n\nModel summary:')
    #x, t, y, cond_img, lsm_cond, topo_cond
    # input_size_summary = (input_channels, image_size[0], image_size[1]), (1,), (1,), (1, image_size[0], image_size[1]), (1, image_size[0], image_size[1]), (1, image_size[0], image_size[1])
    # summary(model, input_size=[input_size_summary], batch_size=batch_size, device=device)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Generating samples...")


    # Set number of samples to generate
    n = n_samples_gen

    # Make a dataloader with batch size equal to n
    eval_dataloader = DataLoader(eval_dataset, batch_size=n, shuffle=False, num_workers=4)

    # Generate samples from evaluation dataloader
    
    for idx, samples in tqdm.tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), position=0):
        
        # Extract the samples
        eval_img, eval_season, eval_cond, eval_lsm, eval_topo, eval_sdf, eval_point = extract_samples(samples, device)


        # Print the shapes and types of the different tensors
        print(f'\n\n\nShape of eval truth image: {eval_img.shape}')
        print(f'Type: {eval_img.dtype}')
        print(f'Shape of eval season: {eval_season.shape}')
        print(f'Type: {eval_season.dtype}')
        print(f'Shape of eval condition: {eval_cond.shape}')
        print(f'Type: {eval_cond.dtype}')
        print(f'Shape of eval lsm: {eval_lsm.shape}')
        print(f'Type: {eval_lsm.dtype}')
        print(f'Shape of eval topo: {eval_topo.shape}')
        print(f'Type: {eval_topo.dtype}')
        if loss_type == 'sdfweighted':
            print(f'Shape of eval sdf: {eval_sdf.shape}')
            print(f'Type: {eval_sdf.dtype}\n\n')


        # Generate images from model
        generated_images = diffusion_utils.sample(n=n,
                                                  model=pipeline.model,
                                                  channels_hr=1,
                                                  y=eval_season,
                                                  cond_img=eval_cond,
                                                  lsm_cond=eval_lsm,
                                                  topo_cond=eval_topo
                                                  )
        generated_images = generated_images.detach().cpu()

        # Stop after first iteration, all samples are generated
        break
    # Define path 
    EVAL_SAMPLES_DIR = '/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_clean/'
    SAVE_PATH = 'samples/Samples__temp__64x64__ERA5_cond_lsm_topo_random__sdfweighted__4_seasons_ValidSplitInTime_9yrs_ZscoreScaled_test/eval_samples/'
    
    # Check if save path exists, if not create it
    if not os.path.exists(EVAL_SAMPLES_DIR + SAVE_PATH):
        os.makedirs(EVAL_SAMPLES_DIR + SAVE_PATH)
        print(f'\n\nCreated directory {EVAL_SAMPLES_DIR + SAVE_PATH}...')
    else:
        print(f'\n\nDirectory {EVAL_SAMPLES_DIR + SAVE_PATH} already exists...')

    # Define save name
    SAVE_NAME =  model_str + '__' + var_str + '__' + im_dim_str + '__' + cond_str + '__' + str(n_samples_gen) + '_samples.npz'

    # Save the generated and corresponding eval images
    print(f'\n\nSaving generated and eval images to {SAVE_PATH}...')
    print(f'Saving as {EVAL_SAMPLES_DIR + SAVE_PATH + "gen_samples__" + SAVE_NAME}')
    np.savez_compressed(EVAL_SAMPLES_DIR + SAVE_PATH + 'gen_samples__' + SAVE_NAME, generated_images)
    np.savez_compressed(EVAL_SAMPLES_DIR + SAVE_PATH + 'eval_samples__' + SAVE_NAME, eval_img)
    np.savez_compressed(EVAL_SAMPLES_DIR + SAVE_PATH + 'lsm_samples__' + SAVE_NAME, eval_lsm)
    np.savez_compressed(EVAL_SAMPLES_DIR + SAVE_PATH + 'cond_samples__' + SAVE_NAME, eval_cond)
    np.savez_compressed(EVAL_SAMPLES_DIR + SAVE_PATH + 'season_samples__' + SAVE_NAME, eval_season)
    np.savez_compressed(EVAL_SAMPLES_DIR + SAVE_PATH + 'point_samples__' + SAVE_NAME, eval_point)






        # fig, axs = plt.subplots(5, n, figsize=(18,8)) # Plotting truth, condition, generated, lsm and topo for n different test images

        # # Loop through the generated samples (and corresponding truth, condition, lsm and topo) and plot
        # for i in range(n):
        #     img_truth = test_img[i].squeeze()
        #     img_cond = test_cond[i].squeeze()
        #     img_gen = generated_image[i].squeeze()
        #     img_lsm = test_lsm[i].squeeze()
        #     img_topo = test_topo[i].squeeze()

        #     image_truth = axs[0, i].imshow(img_truth, cmap='viridis')
        #     axs[0, i].set_title(f'Truth')
        #     axs[0, i].axis('off')
        #     axs[0, i].set_ylim([0, img_truth.shape[0]])
        #     fig.colorbar(image_truth, ax=axs[0, i], fraction=0.046, pad=0.04)
            
        #     image_cond = axs[1, i].imshow(img_cond, cmap='viridis')
        #     axs[1, i].set_title(f'Condition')
        #     axs[1, i].axis('off')
        #     axs[1, i].set_ylim([0, img_cond.shape[0]])
        #     fig.colorbar(image_cond, ax=axs[1, i], fraction=0.046, pad=0.04)

        #     image_gen = axs[2, i].imshow(img_gen, cmap='viridis')
        #     axs[2, i].set_title(f'Generated')
        #     axs[2, i].axis('off')
        #     axs[2, i].set_ylim([0, img_gen.shape[0]])
        #     fig.colorbar(image_gen, ax=axs[2, i], fraction=0.046, pad=0.04)

        #     image_lsm = axs[3, i].imshow(img_lsm, cmap='viridis')
        #     axs[3, i].set_title(f'LSM')
        #     axs[3, i].axis('off')
        #     axs[3, i].set_ylim([0, img_lsm.shape[0]])
        #     fig.colorbar(image_lsm, ax=axs[3, i], fraction=0.046, pad=0.04)

        #     image_topo = axs[4, i].imshow(img_topo, cmap='viridis')
        #     axs[4, i].set_title(f'Topography')
        #     axs[4, i].axis('off')
        #     axs[4, i].set_ylim([0, img_topo.shape[0]])
        #     fig.colorbar(image_topo, ax=axs[4, i], fraction=0.046, pad=0.04)

        # fig.tight_layout()
        # plt.show()








# # Generate samples
# x = torch.randn(n, input_channels, *image_size).to(device)

# # Generate random season labels of batchsize n
# y = torch.randint(0, n_seasons, (n,)).to(device) # 4 seasons, 0-3

# # Sample generated images from model
# generated_images = diffusion_utils.sample(x, pipeline.model, y)
# generated_images = generated_images.detach().cpu()

# # Plot samples
# fig, axs = plt.subplots(1, n, figsize=(18,3))

# for i in range(n):
#     img = generated_images[i].squeeze()
#     image = axs[i].imshow(img, cmap='viridis')
#     axs[i].set_title(f'Season: {y[i].item()}')
#     axs[i].axis('off')
#     fig.colorbar(image, ax=axs[i], fraction=0.046, pad=0.04)
# fig.tight_layout()

# plt.show()

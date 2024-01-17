'''
    Script to run evaluations of trained ddpm models on test DANRA dataset.
'''

import torch
import zarr
import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from modules_DANRA_conditional import *
from diffusion_DANRA_conditional import DiffusionUtils
from training_DANRA_conditional import SDFWeightedMSELoss, TrainingPipeline_ERA5_Condition
from data_DANRA_conditional import preprocess_lsm_topography, DANRA_Dataset_cutouts_ERA5_Zarr


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()


    # Define DANRA data information 
    # Set variable for use
    var = 'temp'#'prcp'# 

    # Set DANRA size string for use in path
    danra_size_str = '589x789'
    danra_size = 64


    # Path to test (full danra, to enable cutouts)
    data_dir_danra_test_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_test.zarr'

    # Path to test ERA5 data, 589x789 (same size as DANRA)
    data_dir_era5_test_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_test.zarr'

    # Create zarr groups
    data_danra_test_zarr = zarr.open_group(data_dir_danra_test_w_cutouts_zarr, mode='r')
    data_era5_test_zarr = zarr.open_group(data_dir_era5_test_zarr, mode='r')

    # Define length of test data set
    n_files_test = len(list(data_danra_test_zarr.keys()))

    # Define data hyperparameters
    input_channels = 1
    output_channels = 1
    n_samples = n_files_test
    cache_size = n_files_test
    image_size = (danra_size, danra_size)
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


    # Define diffusion hyperparameters
    n_timesteps = 800
    beta_min = 1e-4
    beta_max = 0.02
    beta_scheduler = 'linear'

    # Create a test dataset
    test_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_test_w_cutouts_zarr,
                                                data_size = image_size,
                                                n_samples = n_samples,
                                                cache_size = cache_size,
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

    # Define model
    model = DiffusionNet(encoder, decoder)

    # Define diffusion 
    diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, device, beta_scheduler)


    # Define path to model checkpoint
    checkpoint_dir = "/Users/au728490/Documents/PhD_AU/Python_Scripts/ModelCheckpoints/DDPM_DANRA"
    checkpoint_name = "DDPM_conditional_TEST__temp__64x64__lsm_topo_random__simple__4_seasons_ValidSplitInTime_9yrs.pth.tar"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path)
    best_model_path = os.path.join(checkpoint_dir, checkpoint_name)
    best_model_state = torch.load(best_model_path)['network_params']


    # Define the loss function
    lossfunc = SDFWeightedMSELoss(max_land_weight=1.0, min_sea_weight=0.0)
    # Define optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Define training pipeline
    pipeline = TrainingPipeline_ERA5_Condition(model,
                                            lossfunc,
                                            optimizer,
                                            diffusion_utils,
                                            device,
                                            weight_init=True,
                                            sdf_weighted_loss=True
                                            )

    # Load best state model into model
    pipeline.model.load_state_dict(best_model_state)
    pipeline.model.eval()

    print("Generating samples...")

    # Set number of samples to generate
    n = 8

    # Make a dataloader with batch size equal to n
    final_dataloader = DataLoader(test_dataset, batch_size=n, shuffle=False, num_workers=1)

    # Generate samples from final dataloader
    for idx, samples in enumerate(final_dataloader):
        # for i, obj in enumerate(samples):
        #     print(f'\nObject number {i}')
        #     print(obj)
        #     try:
        #         print(f'Shape: {obj.shape}')
        #         print(f'Type: {obj.dtype}')
        #     except:
        #         print(f'Length: {len(obj)}')
        #         print(f'Type: {type(obj)}')
            
        # Get test image, season, condition, lsm and topo
        (test_img, test_season, test_cond), test_lsm, test_topo, sdf_use, _ = samples

        # Generate random fields of same shape as test image and send to device
        x = torch.randn(n, input_channels, *image_size).to(device)
        # Send all other parts of sample to device
        test_season = test_season.to(device)
        test_cond = test_cond.to(torch.float).to(device)
        test_lsm = test_lsm.to(device)
        test_topo = test_topo.to(device)


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

        fig, axs = plt.subplots(5, n, figsize=(18,8)) # Plotting truth, condition, generated, lsm and topo for n different test images

        # Loop through the generated samples (and corresponding truth, condition, lsm and topo) and plot
        for i in range(n):
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

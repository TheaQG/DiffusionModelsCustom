'''
    Tests if input and output shapes are correct for the UNet.

'''

import torch
import zarr
import os


from multiprocessing import freeze_support
from torchinfo import summary
from torch.utils.data import DataLoader

from ..src.unet import *
from ..src.utils import *
from ..src.data_modules import *
from ..src.diffusion_modules import *

if __name__ == '__main__':
    freeze_support()

    var = 'temp'#'prcp'#

    # Set size of DANRA images
    n_danra_size = 64 #32 #256#
    # Set DANRA size string for use in path
    danra_size_str = '589x789'#str(n_danra_size) + 'x' + str(n_danra_size)

    # Path to DANRA data (zarr files), full danra, to enable cutouts
    data_dir_danra_train_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_train.zarr'

    # Path to ERA5 data, 589x789 (same size as DANRA)
    data_dir_era5_train_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_train.zarr'

    # Make zarr groups
    data_danra_train_zarr = zarr.open_group(data_dir_danra_train_w_cutouts_zarr, mode='r')
    data_era5_train_zarr = zarr.open_group(data_dir_era5_train_zarr, mode='r')

    n_files_train = len(list(data_danra_train_zarr.keys()))
    n_samples_train = n_files_train
    cache_size_train = n_samples_train

    # Define data hyperparameters (channels for input and output tensors)
    input_channels = 1
    output_channels = 1

    # Image dimensions
    image_dim = n_danra_size#64#32#64#
    image_size = (image_dim,image_dim)

    n_seasons = 4#12#366#

    scaling = True

    # Define model hyperparameters
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    first_fmap_channels = 64#n_danra_size #
    last_fmap_channels = 512 #2048
    time_embedding = 256


    CUTOUTS = True
    CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]

    PATH_LSM_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
    PATH_TOPO_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

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


    # Define training dataset, with cutouts enabled and data from zarr files
    train_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_train_w_cutouts_zarr, 
                                            data_size = image_size, 
                                            n_samples = n_samples_train, 
                                            cache_size = cache_size_train, 
                                            scale=scaling,
                                            variable=var, 
                                            conditional_seasons=True,
                                            conditional_images=True,
                                            cond_dir_zarr=data_dir_era5_train_zarr, 
                                            n_classes=n_seasons, 
                                            cutouts=CUTOUTS, 
                                            cutout_domains=CUTOUT_DOMAINS,
                                            lsm_full_domain=data_lsm_full,
                                            topo_full_domain=data_topo_full,
                                            sdf_weighted_loss = True
                                            )


    # Define the torch dataloaders for train and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    n_test_samples = 4

    # Define the encoder and decoder from modules_DANRA_downscaling.py
    encoder = Encoder(input_channels, 
                        time_embedding, 
                        cond_on_img=True, 
                        cond_img_dim=(1, image_size[0], image_size[1]), 
                        block_layers=[2, 2, 2, 2], 
                        num_classes=n_seasons,
                        n_heads=image_dim//2
                        )
    decoder = Decoder(last_fmap_channels, 
                        output_channels, 
                        time_embedding, 
                        first_fmap_channels,
                        n_heads= 8#image_dim//2
                        )
    # Define the model from modules_DANRA_downscaling.py
    model = DiffusionNet(encoder, decoder)


    print('\n\n\nTesting model output size...\n')
    n_samples = 4

    test_input = torch.randn(n_samples, input_channels, *image_size).to(device)
    t_test = torch.randint(0, time_embedding, (n_samples,)).to(device)
    y_test = torch.randint(0, n_seasons, (n_samples, ))
    cond_img = torch.randn(n_samples, 1, image_size[0], image_size[1]).to(device)
    lsm_test = torch.ones(n_samples, 1, image_size[0], image_size[1]).to(device)
    topo_test = torch.ones(n_samples, 1, image_size[0], image_size[1]).to(device)

    y_test = y_test.type(torch.int64).to(device)

    print(f'Input shape of test: {test_input.shape}')
    print(f'Shape of time embedding: {t_test.shape}')
    print(f'Shape of season embedding: {y_test.shape}')
    print(f'Shape of condition image: {cond_img.shape}')
    print(f'Shape of lsm condition: {lsm_test.shape}')
    print(f'Shape of topo condition: {topo_test.shape}\n\n')
    test = model(test_input, t_test, y_test, cond_img=cond_img, lsm_cond=lsm_test, topo_cond=topo_test)

    inputs = (
        test_input,
        t_test,
        y_test,
        cond_img,
        lsm_test,
        topo_test
    )


    print(f'Output shape of test: {test.shape}')
 
import zarr 
import random

import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import freeze_support

# Import DANRA dataset class from data_modules.py in src folder
from ..src.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr

# Test the dataset class 
if __name__ == '__main__':
    # Use multiprocessing freeze_support() to avoid RuntimeError:
    freeze_support()


    # Set DANRA variable for use
    var = 'temp'#'prcp'#
    if var == 'temp':
        cmap_name = 'plasma'
        cmap_label = r'$^\circ$C'
    elif var == 'prcp':
        cmap_name = 'inferno'
        cmap_label = 'mm'


    # Set size of DANRA images
    n_danra_size = 128#32#64#
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)
    # Define different samplings
    sample_w_lsm_topo = True
    sample_w_cutouts = True
    sample_w_cond_img = True
    sample_w_cond_season = True

    # Set scaling to true or false
    scaling = True


    # Set paths to zarr data
    data_dir_danra_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789/' + var + '_589x789/zarr_files/test.zarr'
    data_dir_era5_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/' + var + '_589x789/zarr_files/test.zarr'

    # Set path to save figures
    SAVE_FIGS = False
    PATH_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures'

    # Set number of samples and cache size
    danra_w_cutouts_zarr_group = zarr.open_group(data_dir_danra_w_cutouts_zarr, mode='r')
    n_samples = len(list(danra_w_cutouts_zarr_group.keys()))#365
    cache_size = n_samples
    # Set image size
    image_size = (n_danra_size, n_danra_size)
    
    CUTOUTS = True
    CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]
    # DOMAIN_2 = [80, 80+130, 200, 200+450]
    # Set paths to lsm and topo if used
    if sample_w_lsm_topo:
        data_dir_lsm = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
        data_dir_topo = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

        data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
        data_topo = np.flipud(np.load(data_dir_topo)['data'])

        if scaling:
            topo_min, topo_max = -12, 330
            norm_min, norm_max = 0, 1
            
            OldRange = (topo_max - topo_min)
            NewRange = (norm_max - norm_min)

            # Generating the new data based on the given intervals
            data_topo = (((data_topo - topo_min) * NewRange) / OldRange) + norm_min

    
    # Initialize dataset with all options
    dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_danra_w_cutouts_zarr, 
                                    image_size, 
                                    n_samples, 
                                    cache_size, 
                                    variable = var,
                                    shuffle=True, 
                                    cutouts = CUTOUTS, 
                                    cutout_domains = CUTOUT_DOMAINS,
                                    n_samples_w_cutouts = n_samples, 
                                    lsm_full_domain = data_lsm,
                                    topo_full_domain = data_topo,
                                    sdf_weighted_loss = True,
                                    scale=True, 
                                    in_low=0,
                                    in_high=1,
                                    data_min_in=-30,
                                    data_max_in=40,
                                    conditional_seasons=True,
                                    conditional_images=True,
                                    cond_dir_zarr = data_dir_era5_zarr,
                                    n_classes=12
                                    )

    # Check if datasets work
    # Get sample images


    n_samples = 3
    idxs = random.sample(range(0, len(dataset)), n_samples)

    fig, axs = plt.subplots(n_samples, 5, figsize=(10,8))

    
    print('\n\nDataset with all options:\n')
    for i, idx in enumerate(idxs):
        # Dataset with all options
        sample_full = dataset[idx]
        
        im1 = axs[i,0].imshow(sample_full['img'].squeeze(), cmap=cmap_name)
        axs[i,0].invert_yaxis()
        axs[i,0].set_xticks([])
        axs[i,0].set_yticks([])
        fig.colorbar(im1, ax=axs[i,0], fraction=0.046, pad=0.04, label=cmap_label)

        im2 = axs[i,1].imshow(sample_full['img_cond'].squeeze(), cmap=cmap_name)
        axs[i,1].invert_yaxis()
        axs[i,1].set_xticks([])
        axs[i,1].set_yticks([])
        fig.colorbar(im2, ax=axs[i,1], fraction=0.046, pad=0.04, label=cmap_label)

        im3 = axs[i,2].imshow(sample_full['topo'].squeeze(), cmap='terrain')
        axs[i,2].invert_yaxis()
        axs[i,2].set_xticks([])
        axs[i,2].set_yticks([])
        fig.colorbar(im3, ax=axs[i,2], fraction=0.046, pad=0.04)
        
        im4 = axs[i,3].imshow(sample_full['lsm'].squeeze(), cmap='binary')
        axs[i,3].invert_yaxis()
        axs[i,3].set_xticks([])
        axs[i,3].set_yticks([])
        fig.colorbar(im4, ax=axs[i,3], fraction=0.046, pad=0.04)
        
        im5 = axs[i,4].imshow(sample_full['sdf'].squeeze(), cmap='coolwarm')
        axs[i,4].invert_yaxis()
        axs[i,4].set_xticks([])
        axs[i,4].set_yticks([])
        fig.colorbar(im5, ax=axs[i,4], fraction=0.046, pad=0.04)

        for key, value in sample_full.items():
            try:
                print(key, value.shape)
                if key == 'classifier':
                    print(value)
            except:
                print(key, len(value))
    
    fig.tight_layout()

    plt.show()



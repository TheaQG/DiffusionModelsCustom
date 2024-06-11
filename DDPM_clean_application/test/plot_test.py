'''
    Script to test plotting helper functions
'''
import zarr
import torch

import numpy as np

from ..src.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from ..src.plot_utils import plot_sample

if __name__ == '__main__':

    # Set DANRA variable for use
    var = 'temp'#'prcp'#

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
    data_dir_danra_w_cutouts_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789_full/zarr_files/' + var + '_589x789_test.zarr'
    data_dir_era5_zarr = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_test.zarr'

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

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Test plot_sample function
    plot_sample(dataloader, variable=var, n_samples=4, show_figs=True)
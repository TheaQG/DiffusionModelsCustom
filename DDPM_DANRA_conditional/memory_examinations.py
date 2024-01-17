import numpy as np
import torch
from data_DANRA_conditional import *
import sys

if __name__ == '__main__':

    # Set DANRA variable for use
    var = 'prcp'#'temp'#
    # Set size of DANRA images
    n_danra_size = 128#32#64#
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)

    # Set paths to data
    # data_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str
    # data_dir_danra_w_cutouts = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789_full/' + var + '_589x789'
    # data_dir_era5 = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_589x789/' + var + '_589x789'

    data_dir_lsm = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
    data_dir_topo = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

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

    data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
    data_topo = np.flipud(np.load(data_dir_topo)['data'])

    # Initialize dataset

    #dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size, scale=False, conditional=True, n_classes=12)
    dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_danra_w_cutouts_zarr, 
                                    image_size, 
                                    n_samples, 
                                    cache_size, 
                                    variable = var,
                                    cutouts = CUTOUTS, 
                                    shuffle=True, 
                                    cutout_domains = CUTOUT_DOMAINS,
                                    n_samples_w_cutouts = n_samples, 
                                    lsm_full_domain = data_lsm,
                                    topo_full_domain = data_topo,
                                    sdf_weighted_loss = True,
                                    scale=False, 
                                    conditional=True,
                                    cond_dir_zarr = data_dir_era5_zarr,
                                    n_classes=12
                                    )

    # Get sample images

    n_samples = 1
    idx = random.sample(range(0, len(dataset)), n_samples)[0]

    (sample_img, sample_season, sample_cond), sample_lsm, sample_topo, sample_sdf, sample_point = dataset[idx]

    def get_size_of_full_sample(sample_img,
                                sample_season,
                                sample_cond,
                                sample_lsm,
                                sample_topo,
                                sample_sdf,
                                sample_point):
        # Convert to numpy
        sample_img = sample_img.numpy()
        sample_season = sample_season.numpy()
        sample_cond = sample_cond.numpy()
        sample_lsm = sample_lsm.numpy()
        sample_topo = sample_topo.numpy()
        sample_sdf = sample_sdf.numpy()
        sample_point = np.array(sample_point)

        # Get size of each sample
        size_img = sample_img.nbytes
        size_season = sample_season.nbytes
        size_cond = sample_cond.nbytes
        size_lsm = sample_lsm.nbytes
        size_topo = sample_topo.nbytes
        size_sdf = sample_sdf.nbytes
        size_point = sys.getsizeof(sample_point)

        # Get total size of sample
        size_sample = size_img + size_season + size_cond + size_lsm + size_topo + size_sdf + size_point

        return size_sample, size_img, size_season, size_cond, size_lsm, size_topo, size_sdf, size_point

    size_sample, size_img, size_season, size_cond, size_lsm, size_topo, size_sdf, size_point = get_size_of_full_sample(sample_img,
                                                                                                                        sample_season,
                                                                                                                        sample_cond,
                                                                                                                        sample_lsm,
                                                                                                                        sample_topo,
                                                                                                                        sample_sdf,
                                                                                                                        sample_point)

    print(f'Size of full sample: {size_sample} bytes, {size_sample/(1024*1024)} MB')
import zarr
import os
import pysal
from esda.moran import Moran
import torch

import numpy as np
import matplotlib.pyplot as plt

from data_DANRA_conditional import preprocess_lsm_topography_from_data


if __name__ == '__main__':
    print('\n\n')

    var = 'temp'
    loss_type = 'sdfweighted'#'simple'#
    danra_size = 64
    n_seasons = 4#None#4#
    n_samples_gen = 100

    image_dim = danra_size
    im_dim_str = str(image_dim) + 'x' + str(image_dim)

    cond_str = 'ERA5_cond_lsm_topo_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs_ValLoss'  #- 'sdfweighted' '4' seasons
    # HAVE NOT BEEN EVALUATED:    




    # HAVE BEEN EVALUATED:
    #'unconditional_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs' #- 'simple' 'None' seasons
    #'DDPM_unconditional'

    #'ERA5_cond_lsm_topo_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs' #  - 'simple' '4' seasons
    #'DDPM_conditional_ERA5'

    #'uniform_cond_lsm_topo_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs' #- 'sdweighted' '4' seasons
    #'DDPM_conditional_uniform

    #'cond_lsm_topo_only_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs' #- 'sdfweighted' '4' seasons
    #'DDPM_conditional_lsm_topo_only'
    
    #'ERA5_cond_lsm_topo_random__' + loss_type + '__' + str(n_seasons) + '_seasons' + '_ValidSplitInTime_9yrs_ValLoss'  #- 'sdfweighted' '4' seasons
    #'DDPM_conditional_ERA5'


    var_str = var


    model_str = 'DDPM_conditional_ERA5'
    SAVE_PATH = '/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_DANRA_conditional/final_generated_samples/'
    SAVE_NAME =  model_str + '__' + var_str + '__' + im_dim_str + '__' + cond_str + '__' + str(n_samples_gen) + '_samples.npz'
    FIG_PATH = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/DDPM_evaluation/'
    # Load generated images, truth evaluation images and lsm mask for each image
    gen_imgs = np.load(SAVE_PATH + 'gen_samples__' + SAVE_NAME)['arr_0']
    eval_imgs = np.load(SAVE_PATH + 'eval_samples__' + SAVE_NAME)['arr_0']
    lsm_imgs = np.load(SAVE_PATH + 'lsm_samples__' + SAVE_NAME)['arr_0']

    



    # Convert to torch tensors
    gen_imgs = torch.from_numpy(gen_imgs).squeeze()
    eval_imgs = torch.from_numpy(eval_imgs).squeeze()
    lsm_imgs = torch.from_numpy(lsm_imgs).squeeze()

    # Plot example of generated and eval images w/o masking
    plot_idx = np.random.randint(0, len(gen_imgs))

    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    im1 = axs[0].imshow(eval_imgs[plot_idx])
    axs[0].set_ylim([0,eval_imgs[plot_idx].shape[0]])
    axs[0].set_title(f'Evaluation image', fontsize=14)
    # Remove ticks and labels
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    fig.colorbar(im1, ax=axs[0])

    im = axs[1].imshow(gen_imgs[plot_idx])
    axs[1].set_ylim([0,gen_imgs[plot_idx].shape[0]])
    axs[1].set_title(f'Generated image', fontsize=14)
    # Remove ticks and labels
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.colorbar(im, ax=axs[1])
    fig.tight_layout()

#    fig.savefig(FIG_PATH + cond_str + '__example_eval_gen_images.png', dpi=600, bbox_inches='tight')


    # Mask out ocean pixels, set to nan
    for i in range(len(gen_imgs)):
        gen_imgs[i][lsm_imgs[i]==0] = np.nan
        eval_imgs[i][lsm_imgs[i]==0] = np.nan

    # Plot a sample of the generated and eval images
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    im1 = axs[0].imshow(eval_imgs[plot_idx])
    axs[0].set_ylim([0,eval_imgs[plot_idx].shape[0]])
    axs[0].set_title(f'Evaluation image', fontsize=14)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    fig.colorbar(im1, ax=axs[0])

    im = axs[1].imshow(gen_imgs[plot_idx])
    axs[1].set_ylim([0,gen_imgs[plot_idx].shape[0]])
    axs[1].set_title(f'Generated image', fontsize=14)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.colorbar(im, ax=axs[1])
    fig.tight_layout()

#    fig.savefig(FIG_PATH + cond_str + '__example_eval_gen_images_masked.png', dpi=600, bbox_inches='tight')


    # Now evaluate the generated samples
    print("Evaluating samples...")

    # Calculate daily average MAE and RMSE for all samples (average over spatial dimensions) ignoring nans
    mae_daily = torch.abs(gen_imgs - eval_imgs).nanmean(dim=(1,2))
    rmse_daily = torch.sqrt(torch.square(gen_imgs - eval_imgs).nanmean(dim=(1,2)))


    # Calculate total single pixel-wise MAE and RMSE for all samples, no averaging
    # Flatten and concatenate the generated and eval images
    gen_imgs_flat = gen_imgs.flatten()
    eval_imgs_flat = eval_imgs.flatten()

    # Calculate MAE and RMSE for all samples ignoring nans
    mae_all = torch.abs(gen_imgs_flat - eval_imgs_flat)
    rmse_all = torch.sqrt(torch.square(gen_imgs_flat - eval_imgs_flat))

    # Make figure with four plots: MAE daily histogram, RMSE daily histogram, MAE pixel-wise histogram, RMSE pixel-wise histogram
    fig, axs = plt.subplots(2, 1, figsize=(12,6), sharex='col')
    # axs[0,0].hist(mae_daily, bins=50)
    # axs[0,0].set_title(f'MAE daily')
    # # axs[0,0].set_xlabel(f'MAE')
    # # axs[0,0].set_ylabel(f'Count')

    # axs[0].hist(mae_all, bins=70)
    # axs[0].set_title(f'MAE for all pixels')
    # axs[0].set_xlabel(f'MAE')
    # axs[0].set_ylabel(f'Count')

    axs[0].hist(rmse_daily, bins=150, alpha=0.7, label='RMSE daily', edgecolor='k')
    axs[0].set_title(f'RMSE daily', fontsize=16)
    axs[0].tick_params(axis='y', which='major', labelsize=14)
    #axs[0].set_xlabel(f'RMSE')
    axs[0].set_ylabel(f'Count', fontsize=16)

    axs[1].hist(rmse_all, bins=1200, alpha=0.7, label='RMSE all pixels', edgecolor='k')
    axs[1].set_title(f'RMSE for all pixels', fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[1].set_xlabel(f'RMSE', fontsize=16)
    axs[1].set_ylabel(f'Count', fontsize=16)
    axs[1].set_xlim([0, 25])

    fig.tight_layout()
    fig.savefig(FIG_PATH + cond_str + '__RMSE_histograms.png', dpi=600, bbox_inches='tight')


    # Plot the pixel-wise distribution of the generated and eval images
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(gen_imgs.flatten(), bins=3000, alpha=0.5, label='Generated')
    ax.hist(eval_imgs.flatten(), bins=50, alpha=0.5, color='r', label='Eval')
    ax.axvline(x=np.nanmean(eval_imgs.flatten()), color='r', alpha=0.5, linestyle='--', label=f'Eval mean, {np.nanmean(eval_imgs.flatten()):.2f}')
    ax.axvline(x=np.nanmean(gen_imgs.flatten()), color='b', alpha=0.5, linestyle='--', label=f'Generated mean, {np.nanmean(gen_imgs.flatten()):.2f}')
    ax.set_title(f'Distribution of generated and eval images, bias: {np.nanmean(gen_imgs.flatten())-np.nanmean(eval_imgs.flatten()):.2f}', fontsize=14)
    ax.set_xlabel(f'Pixel value', fontsize=14)
    ax.set_ylabel(f'Count', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Set the x-axis limits to 4 sigma around the mean of the eval images
    ax.set_xlim([np.nanmean(eval_imgs.flatten())-4*np.nanstd(eval_imgs.flatten()), np.nanmean(eval_imgs.flatten())+4*np.nanstd(eval_imgs.flatten())])
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_PATH + cond_str + '__pixel_distribution.png', dpi=600, bbox_inches='tight')


    # # Calculate Moran's I for all samples. Plot histogram of Moran's I values
    # morans_I = []
    # for gen_im, ev_im in zip(gen_imgs, eval_imgs):
    #     morans_I.append(Moran(gen_im.flatten(), pysal.weights.lat2W(image_dim, image_dim)).I)
    # morans_I = np.array(morans_I)

    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.hist(morans_I, bins=20)
    # ax.set_title(f'Moran\'s I for all samples, mean: {np.nanmean(morans_I):.2f}')
    # ax.set_xlabel(f'Moran\'s I')
    # ax.set_ylabel(f'Count')
    plt.show()



    # # Get the LSM mask for the area that the generated images are cropped from
    # CUTOUTS = True
    # CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]

    # PATH_LSM_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
    # PATH_TOPO_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

    # data_lsm_full = np.flipud(np.load(PATH_LSM_FULL)['data'])[CUTOUT_DOMAINS[0]:CUTOUT_DOMAINS[1], CUTOUT_DOMAINS[2]:CUTOUT_DOMAINS[3]]
    # data_topo_full = np.flipud(np.load(PATH_TOPO_FULL)['data'])[CUTOUT_DOMAINS[0]:CUTOUT_DOMAINS[1], CUTOUT_DOMAINS[2]:CUTOUT_DOMAINS[3]]

    # # Load the points for the cutouts
    # points_imgs = np.load(SAVE_PATH + 'point_samples__' + SAVE_NAME)['arr_0']





    # NEED TO TAKE IMAGE SHIFT INTO ACCOUNT FOR THE PIXEL-WISE IMAGES 
    # # Calculate pixel-wise MAE and RMSE for all samples (average over temporal dimension)
    # mae_pixel = torch.abs(gen_imgs - eval_imgs).mean(dim=0)
    # rmse_pixel = torch.sqrt(torch.square(gen_imgs - eval_imgs).mean(dim=0))


    # # Plot image of MAE and RMSE for temporal average
    # fig, axs = plt.subplots(1, 2, figsize=(12,4))
    # im1 = axs[0].imshow(mae_pixel)
    # axs[0].set_ylim([0,mae_pixel.shape[0]])
    # axs[0].set_title(f'MAE pixel-wise')
    # fig.colorbar(im1, ax=axs[0])

    # im2 = axs[1].imshow(rmse_pixel)
    # axs[1].set_title(f'RMSE pixel-wise')
    # axs[1].set_ylim([0,rmse_pixel.shape[0]])
    # fig.colorbar(im2, ax=axs[1])



    # # Calculate Pearson correlation coefficient for all samples
    # for gen_im, ev_im in zip(gen_imgs, eval_imgs):
    #     corr = np.ma.corrcoef(np.ma.masked_invalid(gen_im), np.ma.masked_invalid(ev_im))
    #     print(corr)



    plt.show()









# # FID score
# # Heidke/Pierce skill score (precipitation)
# # EV analysis (precipitation)
# # Moran's I (spatial autocorrelation)
# # Bias per pixel (spatial bias) 
# # Bias per image (temporal bias)
# # Bias per pixel per image (spatio-temporal bias)






import os
import random
import datetime
#import torch
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import netCDF4 as nc


def find_rand_points(rect, crop_dim):
    '''
    Function to find random quadrants in a given rectangle
    '''
    x1 = rect[0]
    x2 = rect[1]
    y1 = rect[2]
    y2 = rect[3]

    d = crop_dim 

    l_x = x2 - x1
    l_y = y2 - y1

    a_x = l_x - d
    a_y = l_y - d

    x_rand = random.randint(0, a_x)
    y_rand = random.randint(0,a_y)

    x1_new = x1 + x_rand
    x2_new = x1_new + d
    y1_new = y1 + y_rand 
    y2_new = y1_new + d 

    point = [x1_new, x2_new, y1_new, y2_new]
    return point


if __name__ == '__main__':

    SHOW_FIGS = True
    SAVE_FIGS = False
    # Set DANRA variable for use
    var = 'temp'#'prcp'#
    # Set size of DANRA images
    n_danra_size = 128#128#
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)

    # Set paths to danra and era5 data
    data_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789_full'
    data_dir_era5 = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5/size_128x128'
    
    # Define the date range
    start_date = datetime.date(1991, 1, 1)
    end_date = datetime.date(1991, 12, 31)

    # Calculate total number of days in the date range
    total_days = (end_date - start_date).days

    # Generate a random number within this range
    random_days = random.randint(0, total_days)

    # Add this number of days to the start date to get the random date
    random_date = start_date + datetime.timedelta(days=random_days)

    # Format the random date
    formatted_date = random_date.strftime('%Y%m%d')

    # Get a file list of all the files in the directory
    file_danra_t2m = f't2m_ave_{formatted_date}.nc'
    file_danra_prcp = f'tp_tot_{formatted_date}.nc'

    file_era5_t2m = f'temp_128x128_{formatted_date}.npz'
    file_era5_prcp = f'prcp_128x128_{formatted_date}.npz'
    file_era5_ewvf = f'ewvf_128x128_{formatted_date}.npz'
    file_era5_nwvf = f'nwvf_128x128_{formatted_date}.npz'

    # Load the file
    data_danra_t2m = nc.Dataset(data_dir_danra + '/temp_589x789/' + file_danra_t2m)['t'][0,0,:,:]-273.15
    data_danra_prcp = nc.Dataset(data_dir_danra + '/prcp_589x789/' + file_danra_prcp)['tp'][0,0,:,:]

    data_era5_t2m = np.load(data_dir_era5 + '/temp_128x128/' + file_era5_t2m)['arr_0'] - 273.15    
    data_era5_prcp = np.load(data_dir_era5 + '/prcp_128x128/' + file_era5_prcp)['arr_0']
    data_era5_ewvf = np.load(data_dir_era5 + '/ewvf_128x128/' + file_era5_ewvf)['arr_0']
    data_era5_nwvf = np.load(data_dir_era5 + '/nwvf_128x128/' + file_era5_nwvf)['arr_0']
    

    # Create a Rectangle patch
    shape_danra = data_danra_t2m.shape

    # Set paths to land/sea mask and topography data
    path_lsm = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
    path_topo = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

    # Load lsm and topo data
    lsm = np.flipud(np.load(path_lsm)['data'])
    topo = np.flipud(np.load(path_topo)['data'])
    topo_masked = topo.copy()
    topo_masked[lsm==0] = 0


    # Define domains to draw data from (around DK (DOMAIN_1) and northern coasts of Germany and Netherlands(DOMAIN_2))
    DOMAIN_1 = [340, 340+180, 170, 170+180]    
    DOMAIN_2 = [200, 200+450, 70, 70+130]

    # Find random points in the domains (to select randomly shifted data from the domains)
    points1 = find_rand_points(DOMAIN_1, n_danra_size)
    points2 = find_rand_points(DOMAIN_2, n_danra_size)




    if SHOW_FIGS:
        ####################################
        # Figure: ERA5 data (temperature)  #
        ####################################
        # Set up figure with four subplots
        fig, ax = plt.subplots(1, 4, figsize=(13,4))
        # Plot temperature data with colorbar 
        im_t2m = ax[0].imshow(data_era5_t2m, cmap='coolwarm')
        fig.colorbar(im_t2m, ax=ax[0], fraction=0.046, pad=0.04)
        ax[0].set_title('ERA5 temperature')

        # Plot precipitation data with colorbar 
        im_prcp = ax[1].imshow(data_era5_prcp, cmap='Blues')
        fig.colorbar(im_prcp, ax=ax[1], fraction=0.046, pad=0.04)
        ax[1].set_title('ERA5 precipitation')
        
        # Plot eastward water vapor flux data with colorbar 
        im_ewvf = ax[2].imshow(data_era5_ewvf, cmap='YlOrRd')
        fig.colorbar(im_ewvf, ax=ax[2], fraction=0.046, pad=0.04)
        ax[2].set_title('ERA5 eastward water vapor flux')
        
        # Plot northward water vapor flux data with colorbar
        im_nwvf = ax[3].imshow(data_era5_nwvf, cmap='YlOrRd')
        fig.colorbar(im_nwvf, ax=ax[3], fraction=0.046, pad=0.04)
        ax[3].set_title('ERA5 northward water vapor flux')

        fig.tight_layout()


        ####################################
        # Figure: DANRA data (temperature) #
        # W. domain rectangles and random  # 
        # area selections of size 128x128  #
        ####################################
        
        # Set up figure 
        fig, ax = plt.subplots(figsize=(10,8))

        # Plot data with colorbar (coolwarm). Set y limits to flip the image
        im = ax.imshow(data_danra_t2m, cmap='coolwarm')
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        plt.ylim([0, shape_danra[0]])

        # Create two rectangles to show the domains
        rect1_plot = patches.Rectangle((DOMAIN_1[0], DOMAIN_1[2]), 180, 180, linewidth=1.5, edgecolor='k', facecolor='none')
        rect2_plot = patches.Rectangle((DOMAIN_2[0], DOMAIN_2[2]), 450, 130, linewidth=1.5, edgecolor='k', facecolor='none')

        # Create two rectangles to show the random area selections
        rect3_plot = patches.Rectangle((points1[0], points1[2]), n_danra_size, n_danra_size, linewidth=1.5, linestyle='--', edgecolor='darkgreen', facecolor='none')
        rect4_plot = patches.Rectangle((points2[0], points2[2]), n_danra_size, n_danra_size, linewidth=1.5, linestyle='--', edgecolor='darkgreen', facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect1_plot)
        ax.add_patch(rect2_plot)
        ax.add_patch(rect3_plot)
        ax.add_patch(rect4_plot)
        ax.set_title('DANRA data (temperature)')
        fig.tight_layout()

        if SAVE_FIGS:
            FN_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/fig_DANRA_domain.png'
            fig.savefig(FN_SAVE, dpi=300, bbox_inches='tight')
        
    
        ####################################
        # Figure: DANRA data (precipitati) #
        # W. domain rectangles and random  # 
        # area selections of size 128x128  #
        ####################################

        # Set up figure 
        fig, ax = plt.subplots(figsize=(10,8))

        # Plot data with colorbar (coolwarm). Set y limits to flip the image
        im = ax.imshow(data_danra_prcp, cmap='Blues')
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        plt.ylim([0, shape_danra[0]])

        # Create two rectangles to show the domains
        rect1_plot = patches.Rectangle((DOMAIN_1[0], DOMAIN_1[2]), 180, 180, linewidth=1.5, edgecolor='k', facecolor='none')
        rect2_plot = patches.Rectangle((DOMAIN_2[0], DOMAIN_2[2]), 450, 130, linewidth=1.5, edgecolor='k', facecolor='none')

        # Create two rectangles to show the random area selections
        rect3_plot = patches.Rectangle((points1[0], points1[2]), n_danra_size, n_danra_size, linewidth=1.5, linestyle='--', edgecolor='darkgreen', facecolor='none')
        rect4_plot = patches.Rectangle((points2[0], points2[2]), n_danra_size, n_danra_size, linewidth=1.5, linestyle='--', edgecolor='darkgreen', facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect1_plot)
        ax.add_patch(rect2_plot)
        ax.add_patch(rect3_plot)
        ax.add_patch(rect4_plot)
        ax.set_title('DANRA data (precipitation)')
        fig.tight_layout()


        #########################################
        # Figure: Land/Sea mask and topography  #
        #########################################

        # Define specific lsm colors and colormap
        colors_lsm = ["cornflowerblue", "olivedrab"]  # Sea: Blue, Land: Brown
        lsm_cmap = LinearSegmentedColormap.from_list("land_sea", colors_lsm)

        # Define specific topo colors and colormap (Split colormap at 0m)
        cmap_topo_colors = [(0.0, 'blue'),
                            (-np.min(topo_masked)/np.max(topo_masked), 'white'),
                            (1.0, 'brown')]
        
        # Define specific DK topo colors and colormap (Split colormap at 0m)
        cmap_topo_colors_dk = [(0.0, 'blue'),
                            (-np.min(topo_masked[points2[2]:points2[3], points2[0]:points2[1]])/np.max(topo_masked[points2[2]:points2[3], points2[0]:points2[1]]), 'white'),
                            (1.0, 'brown')]
        terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', cmap_topo_colors)
        terrain_map_dk = colors.LinearSegmentedColormap.from_list('terrain_map_dk', cmap_topo_colors_dk)

        print(cmap_topo_colors)
        print(cmap_topo_colors_dk)
        print(terrain_map)
        # Make a diverging colormap for the topography data with a two-slope norm (split at 0m)
        divnorm = colors.TwoSlopeNorm(vmin=np.min(topo_masked), vcenter=0, vmax=np.max(topo_masked))
        div_norm_dk = colors.TwoSlopeNorm(vmin=np.min(topo_masked[points2[2]:points2[3], points2[0]:points2[1]]), vcenter=0, vmax=np.max(topo_masked[points2[2]:points2[3], points2[0]:points2[1]]))

        # Create rectangles for the domains and random area selections
        rect1_plot = patches.Rectangle((DOMAIN_1[0], DOMAIN_1[2]), 180, 180, linewidth=1.5, edgecolor='k', facecolor='none')
        rect2_plot = patches.Rectangle((DOMAIN_2[0], DOMAIN_2[2]), 450, 130, linewidth=1.5, edgecolor='k', facecolor='none')

        rect3_plot = patches.Rectangle((points1[0], points1[2]), n_danra_size, n_danra_size, linewidth=1.5, linestyle='--', edgecolor='darkgreen', facecolor='none')
        rect4_plot = patches.Rectangle((points2[0], points2[2]), n_danra_size, n_danra_size, linewidth=1.5, linestyle='--', edgecolor='darkgreen', facecolor='none')

        # Set up figure with two subplots
        fig2, ax2 = plt.subplots(1, 2, figsize=(20,8), sharey=True)
        
        # Plot the data with the lsm colormap and add the rectangles
        im2_1 = ax2[0].imshow(lsm, cmap=lsm_cmap)
        fig2.colorbar(im2_1, ax=ax2[0], fraction=0.035, pad=0.04)
        ax2[0].add_patch(rect1_plot)
        ax2[0].add_patch(rect2_plot)
        ax2[0].add_patch(rect3_plot)
        ax2[0].add_patch(rect4_plot)
        
        # Create rectangles for the domains and random area selections
        rect1_plot = patches.Rectangle((DOMAIN_1[0], DOMAIN_1[2]), 180, 180, linewidth=1.5, edgecolor='k', facecolor='none')
        rect2_plot = patches.Rectangle((DOMAIN_2[0], DOMAIN_2[2]), 450, 130, linewidth=1.5, edgecolor='k', facecolor='none')

        rect3_plot = patches.Rectangle((points1[0], points1[2]), n_danra_size, n_danra_size, linewidth=1.5, linestyle='--', edgecolor='darkgreen', facecolor='none')
        rect4_plot = patches.Rectangle((points2[0], points2[2]), n_danra_size, n_danra_size, linewidth=1.5, linestyle='--', edgecolor='darkgreen', facecolor='none')

        # Plot the data with the topography colormap and add the rectangles
        im2_2 = ax2[1].imshow(topo_masked, cmap=terrain_map)
        fig2.colorbar(im2_2, ax=ax2[1], fraction=0.035, pad=0.04)
        ax2[1].add_patch(rect1_plot)
        ax2[1].add_patch(rect2_plot)
        ax2[1].add_patch(rect3_plot)
        ax2[1].add_patch(rect4_plot)
        

        # Set titles and y limits
        ax2[0].set_title('Land/Sea mask')
        ax2[1].set_title('Topography')
        plt.ylim([0, shape_danra[0]])

        fig2.tight_layout()

        #########################################
        # Figure: Random area selection Domain 1#
        #########################################

        # Set up figure with three subplots (data, lsm, topo)
        fig, ax = plt.subplots(1, 4, figsize=(13,4), sharey=True)
        # Plot the data with the DANRA colormap and add the colorbar
        im_data = ax[0].imshow(data_danra_t2m[points1[2]:points1[3], points1[0]:points1[1]], cmap='coolwarm')
        ax[0].set_title('DANRA data (temperature)')
        plt.colorbar(im_data, ax=ax[0], fraction=0.046, pad=0.04)
        # Plot the prcp data with the prcp colormap and add the colorbar
        im_prcp = ax[1].imshow(data_danra_prcp[points1[2]:points1[3], points1[0]:points1[1]], cmap='Blues')
        ax[1].set_title('DANRA data (precipitation)')
        plt.colorbar(im_prcp, ax=ax[1], fraction=0.046, pad=0.04)
        # Plot the lsm data with the lsm colormap 
        im_lsm = ax[2].imshow(lsm[points1[2]:points1[3], points1[0]:points1[1]], cmap=lsm_cmap)
        ax[2].set_title('Land/Sea mask')
        plt.colorbar(im_lsm, ax=ax[2], fraction=0.046, pad=0.04)
        # Plot the topography data with the topography colormap and add the colorbar
        im_topo = ax[3].imshow(topo_masked[points1[2]:points1[3], points1[0]:points1[1]], cmap=terrain_map_dk)#, cmap='BrBG_r', norm=div_norm_dk
        ax[3].set_title('Topography')
        plt.colorbar(im_topo, ax=ax[3], fraction=0.046, pad=0.04)

        fig.tight_layout()
        plt.ylim([0, points1[3]-points1[2]])


        #########################################
        # Figure: Random area selection Domain 2#
        #########################################        
        
        # Set up figure with three subplots (data, lsm, topo)
        fig, ax = plt.subplots(1, 4, figsize=(13,4), sharey=True)
        # Plot the data with the DANRA colormap and add the colorbar
        im_data = ax[0].imshow(data_danra_t2m[points2[2]:points2[3], points2[0]:points2[1]], cmap='coolwarm')
        ax[0].set_title('DANRA data (temperature)')
        plt.colorbar(im_data, ax=ax[0], fraction=0.046, pad=0.04)
        # Plot the prcp data with the prcp colormap and add the colorbar
        im_prcp = ax[1].imshow(data_danra_prcp[points2[2]:points2[3], points2[0]:points2[1]], cmap='Blues')
        ax[1].set_title('DANRA data (precipitation)')
        plt.colorbar(im_prcp, ax=ax[1], fraction=0.046, pad=0.04)
        # Plot the lsm data with the lsm colormap
        im_lsm = ax[2].imshow(lsm[points2[2]:points2[3], points2[0]:points2[1]], cmap=lsm_cmap)
        ax[2].set_title('Land/Sea mask')
        plt.colorbar(im_lsm, ax=ax[2], fraction=0.046, pad=0.04)
        # Plot the topography data with the topography colormap and add the colorbar
        im_topo = ax[3].imshow(topo_masked[points2[2]:points2[3], points2[0]:points2[1]], cmap=terrain_map_dk)
        ax[3].set_title('Topography')
        plt.colorbar(im_topo, ax=ax[3], fraction=0.046, pad=0.04)
        
        fig.tight_layout()
        plt.ylim([0, points2[3]-points2[2]])
        
        plt.show()



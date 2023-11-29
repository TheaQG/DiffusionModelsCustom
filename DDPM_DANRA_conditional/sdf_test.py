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
from scipy.ndimage import distance_transform_edt as distance
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

def generate_sdf(mask):
    # Ensure mask is boolean
    binary_mask = mask > 0 

    # Distance transform for land
    dist_transform_land = distance(binary_mask)

    # Distance transform for sea
    dist_transform_sea = distance(~binary_mask)

    # Combine, with negative sign for sea
    sdf = dist_transform_land - dist_transform_sea

    return sdf

def normalize_sdf(sdf):
    # Find the maximum absolute value in the SDF
    max_abs_val = np.abs(sdf).max()

    # Scale the SDF to the range [-1, 1]
    sdf_normalized = sdf / max_abs_val

    # Shift and scale the range to [0, 1]
    sdf_normalized = (sdf_normalized + 1) / 2

    return sdf_normalized

lsm_data = np.load('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/DANRA/data_lsm/truth_fullDomain/lsm_full.npz')['data']

n_danra_size = 128

# Define domains to draw data from (around DK (DOMAIN_1) and northern coasts of Germany and Netherlands(DOMAIN_2))
DOMAIN_1 = [170, 170+180, 340, 340+180]
DOMAIN_2 = [200, 200+450, 70, 70+130]

# Find random points in the domains (to select randomly shifted data from the domains)
points1 = find_rand_points(DOMAIN_1, n_danra_size)
points2 = find_rand_points(DOMAIN_2, n_danra_size)


lsm_data_small = lsm_data[points1[0]:points1[1], points1[2]:points1[3]]

sdf = generate_sdf(lsm_data_small)

fig, ax = plt.subplots(1,3)
im_lsm = ax[0].imshow(lsm_data_small)
ax[0].set_ylim(ax[0].get_ylim()[::-1])
im_sdf = ax[1].imshow(sdf)
ax[1].set_ylim(ax[1].get_ylim()[::-1])
im_sdf_norm = ax[2].imshow(normalize_sdf(sdf))# + lsm_data_small)
ax[2].set_ylim(ax[2].get_ylim()[::-1])
fig.colorbar(im_sdf_norm, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)
# Show colorbar
fig.colorbar(im_sdf, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)

plt.show()


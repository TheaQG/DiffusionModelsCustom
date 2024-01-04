
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
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

    # Distance transform for sea
    dist_transform_sea = distance(~binary_mask)

    # Set land to 1 and subtract sea distances
    sdf = binary_mask.astype(np.float32) - dist_transform_sea

    return sdf

def normalize_sdf(sdf):
    # Find min and max in the SDF
    min_val = np.min(sdf)
    max_val = np.max(sdf)

    # Normalize the SDF
    sdf_normalized = (sdf - min_val) / (max_val - min_val)

    return sdf_normalized

lsm_data = np.load('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/DANRA/data_lsm/truth_fullDomain/lsm_full.npz')['data']

n_danra_size = 128

# Define domains to draw data from (around DK (DOMAIN_1) and northern coasts of Germany and Netherlands(DOMAIN_2))
DOMAIN_1 = [170, 170+180, 340, 340+180]
DOMAIN_2 = [200, 200+450, 70, 70+130]

# Find random points in the domains (to select randomly shifted data from the domains)
points1 = find_rand_points(DOMAIN_1, n_danra_size)
points2 = find_rand_points(DOMAIN_2, n_danra_size)

# Crop an example from the lsm data
lsm_data_small = lsm_data[points1[0]:points1[1], points1[2]:points1[3]]



# Generate the SDF
sdf = generate_sdf(lsm_data_small)

# Normalize the SDF
sdf_norm = normalize_sdf(sdf)


# Calculate a weights map
max_land_weight = 1.0
min_sea_weight = 0.5

# Convert to torch tensor
sdf_torch = torch.from_numpy(sdf_norm).unsqueeze(0).unsqueeze(0).float()

# Calculate weights, and scale to the desired range
weights = torch.sigmoid(sdf_torch) * (max_land_weight - min_sea_weight) + min_sea_weight




# Plot lsm, sdf and weights
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(lsm_data_small)
ax[0].set_ylim([0,128])

sdf_im = ax[1].imshow(sdf_norm)
ax[1].set_ylim([0,128])
fig.colorbar(sdf_im, ax=ax[1], fraction=0.046, pad=0.04)

weights_im = ax[2].imshow(weights.squeeze().numpy())
ax[2].set_ylim([0,128])
fig.colorbar(sdf_im, ax=ax[2], fraction=0.046, pad=0.04)

fig.tight_layout()
plt.show()
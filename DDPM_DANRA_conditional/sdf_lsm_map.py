import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt 

class SDFWeightedMSELoss(nn.Module):
    '''
        Custom loss function for SDFs.

    '''
    def __init__(self, max_land_weight=1.0, min_sea_weight=0.5):
        super().__init__()
        self.max_land_weight = max_land_weight
        self.min_sea_weight = min_sea_weight
#        self.mse = nn.MSELoss(reduction='none')

    def forward(self, input, target, sdf):
        # Convert SDF to weights, using a sigmoid function (or similar)
        # Scaling can be adjusted to control sharpness of transition
        weights = torch.sigmoid(sdf) * (self.max_land_weight - self.min_sea_weight) + self.min_sea_weight

        # Calculate the squared error
        squared_error = (input - target)**2

        # Apply the weights
        weighted_squared_error = weights * squared_error

        # Return mean of weighted squared error
        return weighted_squared_error.mean()



def generate_sdf(mask):
    # Ensure mask is boolean
    binary_mask = mask > 0 

    # Distance transform for land
    dist_transform_land = distance_transform_edt(binary_mask)

    # Distance transform for sea
    dist_transform_sea = distance_transform_edt(~binary_mask)

    # Combine, with negative sign for sea
    sdf = dist_transform_land - dist_transform_sea

    return sdf
def normalize_sdf(sdf):
    # Find the maximum absolute value in the SDF
    max_abs_val = np.abs(sdf).max()

    # Scale the SDF to the range [-1, 1]
    sdf_norm = sdf / max_abs_val

    # Shift and scale the range to [0, 1]
    sdf_norm = (sdf_norm + 1) / 2

    return sdf_norm

# Read in the lsm file
PATH_LSM = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
lsm_data = np.load(PATH_LSM)['data']

# Generate the SDF
sdf = generate_sdf(lsm_data)

# Normalize the SDF
sdf_norm = normalize_sdf(sdf)

max_land_weight = 1.0
min_sea_weight = 0.

sdf_torch = torch.from_numpy(sdf_norm).unsqueeze(0).unsqueeze(0).float()
weights = torch.sigmoid(sdf_torch) * (max_land_weight - min_sea_weight) + min_sea_weight

# Plot the lsm, sdf and normalized sdf
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(lsm_data)
sdf_im = ax[1].imshow(sdf_norm)
fig.colorbar(sdf_im, ax=ax[1])
weights_im = ax[2].imshow(weights.squeeze().numpy())
fig.colorbar(weights_im, ax=ax[2])

fig.tight_layout()
plt.show()



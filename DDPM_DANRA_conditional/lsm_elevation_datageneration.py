import numpy as np
import matplotlib.pyplot as plt
import torch 
import os



# Import data 
path_lsm_full_ud = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/DANRA/data_lsm/truth_fullDomain/lsm_full.npz'
data_lsm_full_ud = np.load(path_lsm_full_ud)['data']

path_topo_full_ud = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/DANRA/data_topo/truth_fullDomain/topo_full.npz'
data_topo_full_ud = np.load(path_topo_full_ud)['data']

data_lsm_full = np.flipud(data_lsm_full_ud)
data_topo_full = np.flipud(data_topo_full_ud)

print(data_lsm_full.shape)
print(data_topo_full.shape)
# Plot data
fig, ax = plt.subplots(1,2)
ax[0].imshow(data_lsm_full)
ax[1].imshow(data_topo_full)



RES_X = 128
RES_Y = 128
# Set domain
min1 = 73+128 # 95 when 256. 73 when 128
max1 = min1 + RES_X
min2 = 242+128 # 306 when 256. 242 when 128
max2 = min2 + RES_Y


data_lsm_dk_ud = data_lsm_full_ud[min1:max1, min2:max2]
data_topo_dk_ud = data_topo_full_ud[min1:max1, min2:max2]

data_lsm_dk = np.flipud(data_lsm_dk_ud)
data_topo_dk = np.flipud(data_topo_dk_ud)

print(data_lsm_dk.shape)
print(data_topo_dk.shape)
# Plot data
fig, ax = plt.subplots(1,2)
ax[0].imshow(data_lsm_dk)
ax[1].imshow(data_topo_dk)
plt.show()

PATH_SAVE_TOPO_DK = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_DK'
PATH_SAVE_TOPO_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain'
PATH_SAVE_LSM_DK = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_DK'
PATH_SAVE_LSM_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain'

np.savez_compressed(PATH_SAVE_TOPO_DK + '/topo_dk.npz', data=data_topo_dk)
np.savez_compressed(PATH_SAVE_TOPO_FULL + '/topo_full.npz', data=data_topo_full)
np.savez_compressed(PATH_SAVE_LSM_DK + '/lsm_dk.npz', data=data_lsm_dk)
np.savez_compressed(PATH_SAVE_LSM_FULL + '/lsm_full.npz', data=data_lsm_full)
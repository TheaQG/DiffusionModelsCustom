import torch 
import zarr
import os
import argparse

# import netCDF4 as nc
import torch.nn as nn
import numpy as np

def model_summary(model):
    '''
        Simple function to print the model summary
    '''

    print("model_summary")
    print()
    print("Layer_name" + "\t"*7 + "Number of Parameters")
    print("="*100)
    
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t"*10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param =model_parameters[j].numel()+model_parameters[j+1].numel()
            j = j+2
        else:
            param =model_parameters[j].numel()
            j = j+1
        print(str(i) + "\t"*3 + str(param))
        total_params+=param
    print("="*100)
    print(f"Total Params:{total_params}")     


class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.mse = nn.MSELoss()#nn.L1Loss()#$

    def forward(self, predicted, target):
        return self.mse(predicted, target)

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, T=10):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.T = T
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        loss = self.mse(predictions[-1], targets[0])
        
        for t in range(1, self.T):
            loss += self.alpha * self.mse(predictions[t-1], targets[t])
        
        return loss

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
    

def convert_npz_to_zarr(npz_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert DANRA .npz files to zarr files
        
        Parameters:
        -----------
        npz_directory: str
            Directory containing .npz files
        zarr_file: str
            Name of zarr file to be created
    '''
    print(f'\n\nConverting {len(os.listdir(npz_directory))} .npz files to zarr file...')

    # Create zarr group (equivalent to a directory) 
    zarr_group = zarr.open_group(zarr_file, mode='w')
    
    # Make iterator to keep track of progress
    i = 0

    # Loop through all .npz files in the .npz directory
    for npz_file in os.listdir(npz_directory):
        
        # Check if the file is a .npz file (not dir or .DS_Store)
        if npz_file.endswith('.npz'):
            if VERBOSE:
                print(os.path.join(npz_directory, npz_file))
            # Load the .npz file
            npz_data = np.load(os.path.join(npz_directory, npz_file))            

            # Loop through all keys in the .npz file
            for key in npz_data:

                if i == 0:
                    print(f'Key: {key}')
                # Save the data as a zarr array
                zarr_group.array(npz_file.replace('.npz', '') + '/' + key, npz_data[key], chunks=True, dtype=np.float32)
            
            # Print progress if iterator is a multiple of 100
            if (i+1) % 100 == 0:
                print(f'Converted {i+1} files...')
            i += 1


# def convert_nc_to_zarr(nc_directory, zarr_file, VERBOSE=False):
#     '''
#         Function to convert ERA5 .nc files to zarr files
        
#         Parameters:
#         -----------
#         nc_directory: str
#             Directory containing .nc files
#         zarr_file: str
#             Name of zarr file to be created
#     '''
#     print(f'Converting {len(os.listdir(nc_directory))} .nc files to zarr file...')
#     # Create zarr group (equivalent to a directory)
#     zarr_group = zarr.open_group(zarr_file, mode='w')
    
#     # Loop through all .nc files in the .nc directory 
#     for nc_file in os.listdir(nc_directory):
#         # Check if the file is a .nc file (not dir or .DS_Store)
#         if nc_file.endswith('.nc'):
#             if VERBOSE:
#                 print(os.path.join(nc_directory, nc_file))
#             # Load the .nc file
#             nc_data = nc.Dataset(os.path.join(nc_directory, nc_file))
#             # Loop through all variables in the .nc file
#             for var in nc_data.variables:
#                 # Select the data from the variable
#                 data = nc_data[var][:]
#                 # Save the data as a zarr array
#                 zarr_group.array(nc_file.replace('.nc', '') + '/' + var, data, chunks=True, dtype=np.float32)


def extract_samples(samples, device='cpu'):
    '''
        Function to extract samples from dictionary.
        Returns the samples as variables.
        If not in dictionary, returns None.
        Also sends the samples to the device and converts to float.
    '''
    images = None
    seasons = None
    cond_images = None
    lsm = None
    sdf = None
    topo = None

    if 'img' in samples.keys() and samples['img'] is not None:
        images = samples['img'].to(device)
        images = samples['img'].to(torch.float)
    else:
        # Stop if no images (images are required)
        raise ValueError('No images in samples dictionary.')
    
    if 'classifier' in samples.keys() and samples['classifier'] is not None:
        seasons = samples['classifier'].to(device)

    if 'img_cond' in samples.keys() and samples['img_cond'] is not None:
        cond_images = samples['img_cond'].to(device)
        cond_images = samples['img_cond'].to(torch.float)

    if 'lsm' in samples.keys() and samples['lsm'] is not None:
        lsm = samples['lsm'].to(device)
        lsm = samples['lsm'].to(torch.float)
    
    if 'sdf' in samples.keys() and samples['sdf'] is not None:
        sdf = samples['sdf'].to(device)
        sdf = samples['sdf'].to(torch.float)
    
    if 'topo' in samples.keys() and samples['topo'] is not None:
        topo = samples['topo'].to(device)
        topo = samples['topo'].to(torch.float)
    
    if 'point' in samples.keys() and samples['point'] is not None:
        point = samples['point'].to(device)
        point = samples['point'].to(torch.float)
        # Print type of topo
        #print(f'Topo is of type: {topo.dtype}')
    else:
        point = None

    return images, seasons, cond_images, lsm, sdf, topo, point

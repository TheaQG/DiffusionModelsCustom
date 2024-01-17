'''
    File to convert daily ERA5 (.npz)/DANRA (.nc) files to zarr files 
    for better storage and access to prevent memory errors.
'''

import zarr 
import numpy as np
import os 
import netCDF4 as nc

def convert_npz_to_zarr(npz_directory, zarr_file):
    '''
        Function to convert DANRA .npz files to zarr files
        
        Parameters:
        -----------
        npz_directory: str
            Directory containing .npz files
        zarr_file: str
            Name of zarr file to be created
    '''
    # Create zarr group (equivalent to a directory) 
    zarr_group = zarr.open_group(zarr_file, mode='w')

    # Loop through all .npz files in the .npz directory
    for npz_file in os.listdir(npz_directory):
        # Check if the file is a .npz file (not dir or .DS_Store)
        if npz_file.endswith('.npz'):
            print(os.path.join(npz_directory, npz_file))
            # Load the .npz file
            npz_data = np.load(os.path.join(npz_directory, npz_file))
            # Loop through all keys in the .npz file
            for key in npz_data:
                # Save the data as a zarr array
                zarr_group.array(npz_file.replace('.npz', '') + '/' + key, npz_data[key], chunks=True, dtype=np.float32)


def convert_nc_to_zarr(nc_directory, zarr_file):
    '''
        Function to convert ERA5 .nc files to zarr files
        
        Parameters:
        -----------
        nc_directory: str
            Directory containing .nc files
        zarr_file: str
            Name of zarr file to be created
    '''
    # Create zarr group (equivalent to a directory)
    zarr_group = zarr.open_group(zarr_file, mode='w')
    
    # Loop through all .nc files in the .nc directory 
    for nc_file in os.listdir(nc_directory):
        # Check if the file is a .nc file (not dir or .DS_Store)
        if nc_file.endswith('.nc'):
            print(os.path.join(nc_directory, nc_file))
            # Load the .nc file
            nc_data = nc.Dataset(os.path.join(nc_directory, nc_file))
            # Loop through all variables in the .nc file
            for var in nc_data.variables:
                # Select the data from the variable
                data = nc_data[var][:]
                # Save the data as a zarr array
                zarr_group.array(nc_file.replace('.nc', '') + '/' + var, data, chunks=True, dtype=np.float32)

if __name__ == '__main__':
    # Testing the function

    nc_directory = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789_full/temp_589x789_test'
    zarr_file = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789_full/zarr_files/temp_589x789_test.zarr'

    # convert_nc_to_zarr(nc_directory, zarr_file)

    
    # Test loading the zarr file
    zarr_group = zarr.open_group(zarr_file, mode='r')

    files = list(zarr_group.keys())
    idx = np.random.randint(0, len(files))
    
    file_name = files[idx]

    # DANRA temp: 't'
    # DANRA prcp: 'tp'
    # ERA5 temp: 'arr_0'
    # ERA5 prcp: 'arr_0'
    data = zarr_group[file_name]['t'][()]

    print(data)
    
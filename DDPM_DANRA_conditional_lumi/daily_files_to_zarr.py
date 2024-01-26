'''
    File to convert daily ERA5 (.npz)/DANRA (.nc) files to zarr files 
    for better storage and access to prevent memory errors.
'''

import zarr 
import numpy as np
import os 
import netCDF4 as nc

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
    print(f'Converting {len(os.listdir(npz_directory))} .npz files to zarr file...')
    # Create zarr group (equivalent to a directory) 
    zarr_group = zarr.open_group(zarr_file, mode='w')

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
                # Save the data as a zarr array
                zarr_group.array(npz_file.replace('.npz', '') + '/' + key, npz_data[key], chunks=True, dtype=np.float32)


def convert_nc_to_zarr(nc_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert ERA5 .nc files to zarr files
        
        Parameters:
        -----------
        nc_directory: str
            Directory containing .nc files
        zarr_file: str
            Name of zarr file to be created
    '''
    print(f'Converting {len(os.listdir(nc_directory))} .nc files to zarr file...')
    # Create zarr group (equivalent to a directory)
    zarr_group = zarr.open_group(zarr_file, mode='w')
    print('zarr group created')
    
    # Loop through all .nc files in the .nc directory 
    for nc_file in os.listdir(nc_directory):
        # Print the first file name
        if nc_file == os.listdir(nc_directory)[0]:
            print(nc_file)
        # Check if the file is a .nc file (not dir or .DS_Store)
        if nc_file.endswith('.nc'):
            if VERBOSE:
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
    split_list = ['test']#['train', 'valid', 'test']
    var_list = ['temp']#['temp', 'prcp']
    model_list = ['DANRA', 'ERA5']
    lumi_data_dir = '/scratch/project_465000568/data_'
    danra_size_str = '589x789'
    
    for split in split_list:
        for var in var_list:
            for model in model_list:
                print(f'Converting {model} {var} {split} files to zarr...')
                if model == 'DANRA':
                    zarr_file = lumi_data_dir + model + '/size_' + danra_size_str + '_full/zarr_files/' + var + '_' + danra_size_str + '_' + split + '.zarr'
                    data_dir = lumi_data_dir + model + '/size_' + danra_size_str + '_full/' + var + '_' + danra_size_str + '_' + split

                    convert_nc_to_zarr(data_dir, zarr_file, VERBOSE=True)
                elif model == 'ERA5':
                    zarr_file = lumi_data_dir + model + '/size_' + danra_size_str + '/zarr_files/' + var + '_' + danra_size_str + '_' + split + '.zarr'
                    data_dir = lumi_data_dir + model + '/size_' + danra_size_str + '/' + var + '_' + danra_size_str + '_' + split

                    convert_npz_to_zarr(data_dir, zarr_file, VERBOSE=True)
                

                # Test loading the zarr file
                zarr_group = zarr.open_group(zarr_file, mode='r')
                print(zarr_group.info)

    
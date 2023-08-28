'''
    Script containing dataset classes for the DANRA and ERA5 datasets.
    On top of that, it contains a class for the DANRA-ERA5 dataset for conditioning.
    The ERA5 dataset is loaded as a 4-channel image, while the DANRA dataset is loaded as a single-channel image.
    Different transforms can be applied to the datasets.

    ERA5 can be loaded in different domains: 
        - 1024x2048, domain_1024x2048 (Atlantic domain)
        - 589x789, domain_danra (DANRA domain)
        - 128x128, domain_dk (DK small)
        - 256x256, domain_dk (DK large) !!! NOT YET !!!
    
    DANRA can be loaded in different sizes:
        - 128x128
        - 256x256

    TODO:
        - Need to examine the normalization of the datasets as they are generated through CDO commands.
'''

# Import packages 
import os, random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Set DANRA variable for use
    var = 'temp'#'prcp'#
    # Set size of DANRA images
    n_danra_size = 256#128#
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)

    # Set paths to data
    data_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str
    data_dir_era5 = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/era5_multi_channel/era5_4channel/domain_1024x2048/1991'

    # Set data sizes
    data_size_danra = (n_danra_size, n_danra_size)#(128,128)
    data_size_era5 = (1024, 2048)#(128,128)#(256,256)#(589, 789)#

    # Set number of channels for ERA5
    n_chans = 4

    # Set number of samples and cache size
    n_samples = 365
    cache_size = 365

    # Set index for sample image
    idx = 200

    # Set seed for reproducibility
    seed = 50


    ##########################################################
    # Load and examine random samples from the DANRA dataset #
    ##########################################################

    # Load files from directory
    files_danra = os.listdir(data_dir_danra)

    # Remove .DS_Store file if present
    if '.DS_Store' in files_danra:
        files_danra.remove('.DS_Store')

    # Print number of images in directory
    print(f'Number of images: {len(files_danra)}')

    # Plot 16 random samples from directory
    n_rows, n_cols = 4,4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10,10))
    fig.suptitle(f'Sample images from DANRA dataset, shape: {data_size_danra}')

    for i in range(n_rows):
        for j in range(n_cols):
            idx = random.randint(0, len(files_danra))
            img = np.load(os.path.join(data_dir_danra, files_danra[idx]))['data']
            axs[i,j].imshow(img)
            axs[i,j].axis('off')
            axs[i,j].set_title(f'Image {idx}')
    fig.set_tight_layout(True)
    plt.close()
    #plt.show()


class DANRA_Dataset(Dataset):
    '''
        Class for setting the DANRA dataset.
        DANRA data is loaded as a single-channel image - either prcp or temp.
        Different transforms can be applied to the dataset.
    '''
    def __init__(self, data_dir:str, data_size:tuple, n_samples:int=365, cache_size:int=365, seed=42):
        '''
            Initialize the class.
            Input:
                - data_dir: path to directory containing the data
                - data_size: tuple containing the size of the data
                - n_samples: number of samples to load
                - cache_size: number of samples to cache
                - seed: seed for reproducibility
        '''
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.data_size = data_size
        self.cache_size = cache_size
        self.seed = seed

        # Load files from directory
        self.files = sorted(os.listdir(self.data_dir))
        # Remove .DS_Store file if present
        if '.DS_Store' in self.files:
            self.files.remove('.DS_Store')

        # Set seed for reproducibility
        random.seed(self.seed)
        # Sample n_samples from files
        self.files = random.sample(self.files, self.n_samples)

        # Set transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.data_size)#,
            #transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        '''
            Return the length of the dataset.
        '''
        return len(self.files)

    def __getitem__(self, idx:int):
        '''
            Get item from dataset based on index.
            Input:
                - idx: index of item to get
        '''

        # Get file path, join directory and file name
        file_path = os.path.join(self.data_dir, self.files[idx])
        # Load image from file and subtract 273.15 to convert from Kelvin to Celsius
        img = np.load(file_path)['data'] - 273.15

        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)

        return img
    
    def __name__(self, idx:int):
        '''
            Return the name of the file based on index.
            Input:
                - idx: index of item to get
        '''
        return self.files[idx]
    

if __name__ == '__main__':
    # Initialize the DANRA dataset
    dataset_danra = DANRA_Dataset(data_dir_danra, data_size_danra, n_samples, seed)

    # Get sample image
    sample_img_danra = dataset_danra[idx]

    # Print information about sample image
    print(dataset_danra.__name__(idx))
    print('\n#############################')
    print('####### DANRA Dataset #######')
    print('#############################\n')
    print(f'Name of sample image: {dataset_danra.__name__(idx)}')
    print(f'Shape of sample image: {sample_img_danra.shape}')
    print(f'Minimum value in sample image: {sample_img_danra.min()}')
    print(f'Mean value in sample image: {sample_img_danra.mean()}')
    print(f'Maximum value in sample image: {sample_img_danra.max()}')

    # Plot sample image
    fig, ax = plt.subplots()
    fig.suptitle(f'Sample image from DANRA dataset, shape: {sample_img_danra.shape}')
    im = ax.imshow(sample_img_danra.permute(1,2,0))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')







class ERA5_Dataset(Dataset):
    '''
        Class for setting the ERA5 dataset.
        ERA5 data is loaded as a 4-channel image.
        Different transforms can be applied to the dataset.
    '''
    def __init__(self, data_dir:str, data_size:tuple, n_samples:int=365, cache_size:int=365, seed=42):
        '''
            Initialize the class.
            Input:
                - data_dir: path to directory containing the data
                - data_size: tuple containing the size of the data
                - n_samples: number of samples to load
                - cache_size: number of samples to cache
                - seed: seed for reproducibility
        '''
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.data_size = data_size
        self.cache_size = cache_size
        self.seed = seed

        # Load files from directory
        self.files = sorted(os.listdir(self.data_dir))
        # Remove .DS_Store file if present
        if '.DS_Store' in self.files:
            self.files.remove('.DS_Store')

        # Set seed for reproducibility
        random.seed(self.seed)
        # Sample n_samples from files
        self.files = random.sample(self.files, self.n_samples)

        # Set transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.data_size)#,
            #transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        '''
            Return the length of the dataset.
        '''
        return len(self.files)

    def __getitem__(self, idx:int):
        '''
            Get item from dataset based on index.
            Input:
                - idx: index of item to get
        '''

        # Get file path, join directory and file name
        file_path = os.path.join(self.data_dir, self.files[idx])
        # Load image from file, transpose to get channels first
        img = np.transpose(np.load(file_path)['arr_0'], (1,2,0))

        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)


        return img
    
    def __name__(self, idx:int):
        '''
            Return the name of the file based on index.
            Input:
                - idx: index of item to get
        '''

        return self.files[idx]

if __name__ == '__main__':
    # Initialize the ERA5 dataset
    dataset_era5 = ERA5_Dataset(data_dir_era5, data_size_era5, n_samples, seed)

    # Get sample image
    sample_img_era5 = dataset_era5[idx]

    # Print information about sample image
    print('\n#############################')
    print('####### ERA5 Dataset #######')
    print('#############################\n')
    print(f'Name of sample image: {dataset_era5.__name__(idx)}')
    print(f'Shape of sample image: {sample_img_era5.shape}')
    print(f'Minimum value in sample image: {sample_img_era5.min()}')
    print(f'Mean value in sample image: {sample_img_era5.mean()}')
    print(f'Maximum value in sample image: {sample_img_era5.max()}')


    # Plot sample image, one channel at a time with colorbar
    fig, axs = plt.subplots(np.shape(sample_img_era5)[0], 1, figsize=(6,10))
    fig.suptitle(f'Sample image from ERA5 dataset, shape: {sample_img_era5.shape}')
    for i, ax in enumerate(axs.flatten()):
        im = ax.imshow(sample_img_era5[i,:,:], cmap='viridis')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        
    fig.tight_layout()






class DANRA_ERA5_Dataset(Dataset):
    '''
        Class for setting the DANRA-ERA5 dataset for conditioning.
        DANRA is considered the target dataset and ERA5 the conditional dataset.
        The ERA5 dataset is loaded as a 4-channel image, while the DANRA dataset is loaded as a single-channel image.
        Different transforms can be applied to the datasets.
    '''
    def __init__(self, era5_data_dir, era5_data_size, danra_data_dir, danra_data_size, n_samples:int=365, cache_size:int=365, seed=42):
        '''
            Initialize the class.
            Input:
                - era5_data_dir: path to directory containing the ERA5 data
                - era5_data_size: tuple containing the size of the ERA5 data
                - danra_data_dir: path to directory containing the DANRA data
                - danra_data_size: tuple containing the size of the DANRA data
                - n_samples: number of samples to load
                - cache_size: number of samples to cache
                - seed: seed for reproducibility
        '''
        self.era5_data_dir = era5_data_dir
        self.era5_data_size = era5_data_size
        self.danra_data_dir = danra_data_dir
        self.danra_data_size = danra_data_size
        self.n_samples = n_samples
        self.cache_size = cache_size
        self.seed = seed

        # Load files from ERA5 directory and sort
        self.era5_files = sorted(os.listdir(self.era5_data_dir))
        # Remove .DS_Store file if present
        if '.DS_Store' in self.era5_files:
            self.era5_files.remove('.DS_Store')

        # Sample n_samples random from files with seed for reproducibility
        random.seed(self.seed)
        self.era5_files = random.sample(self.era5_files, self.n_samples)

        # Load files from DANRA directory and sort
        self.danra_files = sorted(os.listdir(self.danra_data_dir))
        # Remove .DS_Store file if present
        if '.DS_Store' in self.danra_files:
            self.danra_files.remove('.DS_Store')

        # Sample n_samples random from files with seed for reproducibility
        random.seed(self.seed)
        self.danra_files = random.sample(self.danra_files, self.n_samples)

        # Set ERA5 transforms
        self.era5_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.era5_data_size)#,
            #transforms.Normalize((0.5,), (0.5,))
        ])

        # Set DANRA transforms
        self.danra_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.danra_data_size)#,
            #transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        '''
            Return the length of the dataset.
        '''
        return len(self.era5_files)
    
    def __getitem__(self, idx:int):
        '''
            Get items from dataset based on index.
            Input:
                - idx: index of items to get
        '''
        # Get file paths, join directory and file names
        era5_file_path = os.path.join(self.era5_data_dir, self.era5_files[idx])
        danra_file_path = os.path.join(self.danra_data_dir, self.danra_files[idx])

        # Load images from files, transpose to get channels first for ERA5, subtract 273.15 to convert from Kelvin to Celsius for DANRA
        era5_img = np.transpose(np.load(era5_file_path)['arr_0'], (1,2,0))
        danra_img = np.load(danra_file_path)['data'] - 273.15

        # Apply transforms if any
        if self.era5_transforms:
            era5_img = self.era5_transforms(era5_img)
        if self.danra_transforms:
            danra_img = self.danra_transforms(danra_img)
    
        return era5_img, danra_img
    
    def __name__(self, idx:int):
        '''
            Return the names of the files based on index.
        '''
        return self.era5_files[idx], self.danra_files[idx]


if __name__ == '__main__':
    # Initialize the DANRA-ERA5 dataset
    dataset_dan_era = DANRA_ERA5_Dataset(data_dir_era5, data_size_era5, data_dir_danra, data_size_danra, n_samples, seed)

    # Get sample images from dataset
    sample_img_era5_d, sample_img_danra_e = dataset_dan_era[idx]
    # Print information about sample images
    print(f'\nName of sample image: {dataset_dan_era.__name__(idx)}')


    # Plot sample images of DANRA and ERA5 in same figure,
    # ERA5 images are plotted in a 2x2 grid, DANRA image is plotted in the rightmost column
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10,5))
    gs = GridSpec(nrows=2,ncols=3, figure=fig)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    axs = [ax1, ax2, ax3, ax4]

    for i, ax in enumerate(axs):
        im = ax.imshow(sample_img_era5[i,:,:], cmap='viridis')
        fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
        ax.axis('off')
        

    ax5 = fig.add_subplot(gs[:,2])
    im = ax5.imshow(sample_img_danra_e.permute(1,2,0), cmap='viridis')
    fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    ax5.set_title(f'DANRA sample image, {var}')
    ax5.axis('off')
    fig.suptitle('ERA5 sample image', ha='left', x=0.26, y=0.95)
    fig.tight_layout()
    # print('\n#############################')
    # print('####### ERA5 Dataset #######')
    # print('#############################\n')
    # print(f'Shape of sample image: {sample_img.shape}')
    # print(f'Minimum value in sample image: {sample_img.min()}')
    # print(f'Mean value in sample image: {sample_img.mean()}')
    # print(f'Maximum value in sample image: {sample_img.max()}')


    # fig, axs = plt.subplots(np.shape(sample_img)[0], 1, figsize=(6,10))
    # for i, ax in enumerate(axs.flatten()):
    #     ax.imshow(sample_img[i,:,:])
    #     ax.axis('off')
    # fig.tight_layout()
    #plt.show()
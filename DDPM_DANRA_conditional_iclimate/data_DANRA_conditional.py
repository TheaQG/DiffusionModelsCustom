"""
    Script for generating a pytorch dataset for the DANRA data.
    The dataset is loaded as a single-channel image - either prcp or temp.
    Different transforms are be applied to the dataset.
    A custom transform is used to scale the data to a new interval.
"""
# Import libraries and modules 
import os, random, tqdm, torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Manager as SharedMemoryManager

from multiprocessing import freeze_support


def preprocess_lsm_topography(lsm_path, topo_path, target_size):
    '''
        Preprocess the lsm and topography data.
        Function loads the data, converts it to tensors, normalizes the topography data to [0, 1] interval,
        and upscales the data to match the target size.

        Input:
            - lsm_path: path to lsm data
            - topo_path: path to topography data
            - target_size: tuple containing the target size of the data
    '''
    # 1. Load the Data
    lsm_data = np.load(lsm_path)['data']
    topo_data = np.load(topo_path)['data']
    
    # 2. Convert to Tensors
    lsm_tensor = torch.tensor(lsm_data).float().unsqueeze(0)  # Add channel dimension
    topo_tensor = torch.tensor(topo_data).float().unsqueeze(0)
    
    # 3. Normalize Topography to [0, 1] interval
    topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    
    # 4. Upscale the Fields to match target size
    resize_transform = transforms.Resize(target_size, antialias=True)
    lsm_tensor = resize_transform(lsm_tensor)
    topo_tensor = resize_transform(topo_tensor)
    
    return lsm_tensor, topo_tensor




def determine_season(filename):
    '''
        Determine the season based on the filename, format _YYYYMMDD.npz.
    '''

    # Extract the month from the filename
    month = int(filename.replace('.npz','')[-4:-2]) # Extracting MM from YYYYMMDD
    # Determine season based on month
    if month in [3, 4, 5]:
        return 0
    elif month in [6, 7, 8]:
        return 1
    elif month in [9, 10, 11]:
        return 2
    else:
        return 3

# Define custom transform
class Scale(object):
    '''
        Class for scaling the data to a new interval. 
        The data is scaled to the interval [in_low, in_high].
        The data is assumed to be in the interval [data_min_in, data_max_in].
    '''
    def __init__(self, in_low, in_high, data_min_in, data_max_in):
        '''
            Initialize the class.
            Input:
                - in_low: lower bound of new interval
                - in_high: upper bound of new interval
                - data_min_in: lower bound of data interval
                - data_max_in: upper bound of data interval
        '''
        self.in_low = in_low
        self.in_high = in_high
        self.data_min_in = data_min_in 
        self.data_max_in = data_max_in

    def __call__(self, sample):
        '''
            Call function for the class - scales the data to the new interval.
            Input:
                - sample: datasample to scale to new interval
        '''
        data = sample
        OldRange = (self.data_max_in - self.data_min_in)
        NewRange = (self.in_high - self.in_low)

        # Generating the new data based on the given intervals
        DataNew = (((data - self.data_min_in) * NewRange) / OldRange) + self.in_low

        return DataNew

class DANRA_Dataset(Dataset):
    '''
        Class for setting the DANRA dataset.
        DANRA data is loaded as a single-channel image - either prcp or temp.
        Different transforms are applied to the dataset:
            - ToTensor: converts the data to a tensor
            - Resize: resizes the data to defined size (not crop but resize)
            - Scale: scales the data to a new interval

    '''
    def __init__(self, data_dir:str, data_size:tuple, n_samples:int=365, cache_size:int=365, scale=True,
                 in_low=-1, in_high=1, data_min_in=-30, data_max_in=30):
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
        self.in_low = in_low
        self.in_high = in_high
        self.data_min_in = data_min_in
        self.data_max_in = data_max_in

        # Load files from directory
        self.files = sorted(os.listdir(self.data_dir))
        # Remove .DS_Store file if present
        if '.DS_Store' in self.files:
            self.files.remove('.DS_Store')

        # Sample n_samples from files
        self.files = random.sample(self.files, self.n_samples)
        
        # Set cache for data loading - if cache_size is 0, no caching is used
        self.cache = multiprocessing.Manager().dict()
        # #self.cache = SharedMemoryManager().dict()
        
        # Set transforms
        if scale:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.data_size, antialias=True),
                Scale(self.in_low, self.in_high, self.data_min_in, self.data_max_in)
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.data_size, antialias=True)
                ])

    def __len__(self):
        '''
            Return the length of the dataset.
        '''
        return len(self.files)

    def _addToCache(self, idx:int, data:torch.Tensor):
        '''
            Add item to cache. 
            If cache is full, remove random item from cache.
            Input:
                - idx: index of item to add to cache
                - data: data to add to cache
        '''
        # If cache_size is 0, no caching is used
        if self.cache_size > 0:
            # If cache is full, remove random item from cache
            if len(self.cache) >= self.cache_size:
                # Get keys from cache
                keys = list(self.cache.keys())
                # Select random key to remove
                key_to_remove = random.choice(keys)
                # Remove key from cache
                self.cache.pop(key_to_remove)
            # Add data to cache
            self.cache[idx] = data

    def __getitem__(self, idx:int):
        '''
            Get item from dataset based on index.
            Input:
                - idx: index of item to get
        '''

        # Get file path, join directory and file name
        file_path = os.path.join(self.data_dir, self.files[idx])

        # Extract season from filename
        season = determine_season(self.files[idx])
        season = torch.tensor(season)

        # Load image from file and subtract 273.15 to convert from Kelvin to Celsius
        with np.load(file_path) as data:
            img = data['data'] - 273.15
        

        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)

        sample = (img, season)

        # Add item to cache
        self._addToCache(idx, sample)

        return sample
    
    def __name__(self, idx:int):
        '''
            Return the name of the file based on index.
            Input:
                - idx: index of item to get
        '''
        return self.files[idx]



# Test the dataset class 
if __name__ == '__main__':
    # Use multiprocessing freeze_support() to avoid RuntimeError:
    freeze_support()


    # Set DANRA variable for use
    var = 'temp'#'prcp'#
    # Set size of DANRA images
    n_danra_size = 128#128#
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)

    # Set paths to data
    data_dir_danra = '/home/theaqg/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str

    # Set path to save figures
    SAVE_FIGS = False
    PATH_SAVE = '/home/theaqg/Figures'

    # Set number of samples and cache size
    n_samples = 365
    cache_size = 365
    # Set image size
    image_size = (n_danra_size, n_danra_size)
    


    # Initialize dataset
    dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size, scale=True, data_min_in=-10, data_max_in=30)

    # Get sample images
    
    n_samples = 4
    idxs = random.sample(range(0, len(dataset)), n_samples)
    
    # Plot sample image with colorbar
    fig, axs = plt.subplots(1, n_samples, figsize=(15, 4))
    for idx, ax in enumerate(axs.flatten()):

        sample_img, sample_season = dataset[idx]
        sample_name = dataset.__name__(idx)

        # Print information about sample image
        print(f'\n\nshape: {sample_img.shape}')
        print(f'season: {sample_season}')
        print(f'min pixel value: {sample_img.min()}')
        print(f'mean pixel value: {sample_img.mean()}')
        print(f'max pixel value: {sample_img.max()}\n')

        ax.axis('off')
        ax.set_title(sample_name.replace('.npz','') + ', \nseason: ' + str(sample_season))

        img = sample_img.squeeze()
        image = ax.imshow(img, cmap='viridis', vmin=-1, vmax=1)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    
    fig.set_tight_layout(True)
    plt.show()



    
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



class DateFromFile:
    def __init__(self, filename):
        self.filename = filename
        self.year = int(self.filename[-12:-8])
        self.month = int(self.filename[-8:-6])
        self.day = int(self.filename[-6:-4])

    def determine_season(self):
        # Determine season based on month
        if self.month in [3, 4, 5]:
            return 0
        elif self.month in [6, 7, 8]:
            return 1
        elif self.month in [9, 10, 11]:
            return 2
        else:
            return 3

    def determine_month(self):
        # Returns the month as an integer in the interval [0, 11]
        return self.month - 1

    @staticmethod
    def is_leap_year(year):
        """Check if a year is a leap year"""
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return True
        return False

    def determine_day(self):
        # Days in month for common years and leap years
        days_in_month_common = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_in_month_leap = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Determine if the year is a leap year
        if self.is_leap_year(self.year):
            days_in_month = days_in_month_leap
        else:
            days_in_month = days_in_month_common

        # Compute the day of the year
        day_of_year = sum(days_in_month[:self.month]) + self.day - 1  # "-1" because if it's January 1st, it's the 0th day of the year
        return day_of_year



# def determine_season(filename):
#     '''
#         Determine the season based on the filename, format _YYYYMMDD.npz.
#     '''

#     # Extract the month from the filename
#     month = int(filename.replace('.npz','')[-4:-2]) # Extracting MM from YYYYMMDD
#     # Determine season based on month
#     if month in [3, 4, 5]:
#         return 0
#     elif month in [6, 7, 8]:
#         return 1
#     elif month in [9, 10, 11]:
#         return 2
#     else:
#         return 3
    
# def determine_month(filename):
#     '''
#         Determine the month based on the filename, format _YYYYMMDD.npz.
#         Returns the month as an integer in the interval [0, 11]
#     '''

#     # Extract the month from the filename
#     month = int(filename.replace('.npz','')[-4:-2]) - 1 # Extracting MM from YYYYMMDD
    
#     return month


# def is_leap_year(year):
#     """Check if a year is a leap year"""
#     if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
#         return True
#     return False


# def day_of_year(filename):
#     '''
#         Determine the day of the year based on the filename, format _YYYYMMDD.npz.
#         Returns the day of the year as an integer in the interval [0, 364]
#     '''
    
#     # Days in month for common years and leap years
#     days_in_month_common = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#     days_in_month_leap = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
#     # Extract year, month, and day from the filename
#     year = int(filename[-12:-8])  # Extracting YYYY from _YYYYMMDD
#     month = int(filename[-6:-4])  # Extracting MM from YYYYMMDD
#     day = int(filename[-8:-6])   # Extracting DD from YYYYMMDD
    
#     # Determine if the year is a leap year
#     if is_leap_year(year):
#         days_in_month = days_in_month_leap
#     else:
#         days_in_month = days_in_month_common
    
#     # Compute the day of the year
#     day_of_year = sum(days_in_month[:month]) + day - 1  # "-1" because if it's January 1st, it's the 0th day of the year
    
#     return day_of_year





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
                 in_low=-1, in_high=1, data_min_in=-30, data_max_in=30, conditional=True, n_classes=4):
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
        self.conditional = conditional
        self.n_classes = n_classes

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
        file_name = self.files[idx]
        
        if self.conditional:
            if self.n_classes == 4:

                # Determine class from filename
                dateObj = DateFromFile(file_name)
                classifier = dateObj.determine_season()
            
            elif self.n_classes == 12:
                # Determine class from filename
                dateObj = DateFromFile(file_name)
                classifier = dateObj.determine_month()

            elif self.n_classes == 365:
                # Determine class from filename
                dateObj = DateFromFile(file_name)
                classifier = dateObj.determine_day()

            else:
                raise ValueError('n_classes must be 4, 12 or 365')
            
            # # Convert classifier to one-hot encoding
            # class_idx = torch.tensor(classifier)
            # classifier = torch.zeros(self.n_classes)
            # classifier[class_idx] = 1            

            classifier = torch.tensor(classifier)
        
        elif not self.conditional:
            # Set classifier to None
            classifier = None
        else:
            raise ValueError('conditional must be True or False')



            

        # Load image from file and subtract 273.15 to convert from Kelvin to Celsius
        with np.load(file_path) as data:
            img = data['data'] - 273.15
        

        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)

        if self.conditional:
            # Return sample image and classifier
            sample = (img, classifier)
        else:
            # Return sample image
            sample = (img)

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
    data_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str

    # Set path to save figures
    SAVE_FIGS = False
    PATH_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures'

    # Set number of samples and cache size
    n_samples = 365
    cache_size = 365
    # Set image size
    image_size = (n_danra_size, n_danra_size)
    


    # Initialize dataset
    dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size, scale=False, conditional=True, n_classes=12)

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
        ax.set_title(sample_name.replace('.npz','') + ', \nclass: ' + str(sample_season))

        img = sample_img.squeeze()
        image = ax.imshow(img, cmap='viridis')
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    
    fig.set_tight_layout(True)
    plt.show()



    
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
import netCDF4 as nc
from multiprocessing import freeze_support


def preprocess_lsm_topography(lsm_path, topo_path, target_size, scale=False, flip=False):
    '''
        Preprocess the lsm and topography data.
        Function loads the data, converts it to tensors, normalizes the topography data to [0, 1] interval,
        and upscales the data to match the target size.

        Input:
            - lsm_path: path to lsm data
            - topo_path: path to topography data
            - target_size: tuple containing the target size of the data
    '''
    # 1. Load the Data and flip upside down if flip=True
    if flip:
        lsm_data = np.flipud(np.load(lsm_path)['data']).copy() # Copy to avoid negative strides
        topo_data = np.flipud(np.load(topo_path)['data']).copy() # Copy to avoid negative strides
        
    else:
        lsm_data = np.load(lsm_path)['data']
        topo_data = np.load(topo_path)['data']
    print(lsm_data.shape)

    # 2. Convert to Tensors
    lsm_tensor = torch.tensor(lsm_data).float().unsqueeze(0)  # Add channel dimension
    topo_tensor = torch.tensor(topo_data).float().unsqueeze(0)
    
    if scale:
        # 3. Normalize Topography to [0, 1] interval
        topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    
    # 4. Upscale the Fields to match target size
    resize_transform = transforms.Resize(target_size, antialias=True)
    lsm_tensor = resize_transform(lsm_tensor)
    topo_tensor = resize_transform(topo_tensor)
    
    return lsm_tensor, topo_tensor

def preprocess_lsm_topography_from_data(lsm_data, topo_data, target_size, scale=True):
    '''
        Preprocess the lsm and topography data.
        Function loads the data, converts it to tensors, normalizes the topography data to[0, 1] interval (if scale=True)),
        and upscales the data to match the target size.

        Input:
            - lsm_data: lsm data
            - topo_data: topography data
            - target_size: tuple containing the target size of the data
            - scale: whether to scale the topography data to [0, 1] interval
    '''    
    # 1. Convert to Tensors
    lsm_tensor = torch.tensor(lsm_data).float().unsqueeze(0)  # Add channel dimension
    topo_tensor = torch.tensor(topo_data).float().unsqueeze(0)
    
    if scale:
        # 2. Normalize Topography to [0, 1] interval
        topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    
    # 3. Upscale the Fields to match target size
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


class DateFromFile_nc:
    def __init__(self, filename):
        self.filename = filename
        self.year = int(self.filename[-11:-7])
        self.month = int(self.filename[-7:-5])
        self.day = int(self.filename[-5:-3])

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
    def __init__(self, data_dir:str, data_size:tuple, n_samples:int=365, cache_size:int=365, scale:bool=True, shuffle:bool=False,in_low=-1, in_high=1, data_min_in=-30, data_max_in=30, conditional:bool=True, n_classes:int=4):
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
        self.scale = scale
        self.shuffle = shuffle
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

        # Sample n_samples from files, either randomly or sequentially
        if self.shuffle:
            self.files = random.sample(self.files, self.n_samples)
        else:
            self.files = self.files[0:n_samples]        
        
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

            elif self.n_classes == 366:
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

class DANRA_Dataset_cutouts(Dataset):
    '''
    
    '''
    def __init__(self, 
                data_dir:str,                       # Path to data
                data_size:tuple,                    # Size of data (2D image, tuple)
                n_samples:int=365,                  # Number of samples to load
                cache_size:int=365,                 # Number of samples to cache
                shuffle=False,                      # Whether to shuffle data (or load sequentially)
                cutouts = False,                    # Whether to use cutouts 
                cutout_domains = None,              # Domains to use for cutouts
                n_samples_w_cutouts = None,         # Number of samples to load with cutouts (can be greater than n_samples)
                lsm_full_domain = None,             # Land-sea mask of full domain
                topo_full_domain = None,            # Topography of full domain
                scale=True,                         # Whether to scale data to new interval
                in_low=-1,                          # Lower bound of new interval
                in_high=1,                          # Upper bound of new interval
                data_min_in=-30,                    # Lower bound of data interval
                data_max_in=30,                     # Upper bound of data interval
                conditional=True,                   # Whether to use conditional sampling
                n_classes=4                         # Number of classes for conditional sampling
                ):                       
        '''n_samples_w_cutouts

        '''
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.data_size = data_size
        self.cache_size = cache_size
        self.scale = scale
        self.shuffle = shuffle
        self.cutouts = cutouts
        self.cutout_domains = cutout_domains
        if n_samples_w_cutouts is None:
            self.n_samples_w_cutouts = self.n_samples
        else:
            self.n_samples_w_cutouts = n_samples_w_cutouts
        self.lsm_full_domain = lsm_full_domain
        self.topo_full_domain = topo_full_domain
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

        # Sample n_samples from files, either randomly or sequentially
        if self.cutouts == False:
            if self.shuffle:
                self.files = random.sample(self.files, self.n_samples)
            else:
                self.files = self.files[0:n_samples]
        else:
            if self.shuffle:
                self.files = random.sample(self.files, self.n_samples_w_cutouts)
            else:
                n_individual_samples = len(self.files)
                factor = int(np.ceil(self.n_samples_w_cutouts/n_individual_samples))
                self.files = self.files*factor
                self.files = self.files[0:self.n_samples_w_cutouts]
        
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
                dateObj = DateFromFile_nc(file_name)
                classifier = dateObj.determine_season()
            
            elif self.n_classes == 12:
                # Determine class from filename
                dateObj = DateFromFile_nc(file_name)
                classifier = dateObj.determine_month()

            elif self.n_classes == 366:
                # Determine class from filename
                dateObj = DateFromFile_nc(file_name)
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
        with nc.Dataset(file_path) as data:
            img = data['t'][0,0,:,:] - 273.15
        
        if self.cutouts:
            # Get random point
            point = find_rand_points(self.cutout_domains, 128)
            # Crop image, lsm and topo
            img = img[point[0]:point[1], point[2]:point[3]]
            lsm_use = self.lsm_full_domain[point[0]:point[1], point[2]:point[3]]
            topo_use = self.topo_full_domain[point[0]:point[1], point[2]:point[3]]
        else:
            point = None
        
        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)

            if self.cutouts:
                lsm_use = self.transforms(lsm_use)
                topo_use = self.transforms(topo_use)

        if self.conditional:
            # Return sample image and classifier
            sample = (img, classifier)
        else:
            # Return sample image
            sample = (img)
        
        # Add item to cache
        self._addToCache(idx, sample)

        
        if self.cutouts:
            # Return sample image and classifier and random point for cropping (lsm and topo)
            return sample, lsm_use, topo_use, point
        else:
            # Return sample image and classifier only
            return sample
    
    def __name__(self, idx:int):
        '''
            Return the name of the file based on index.
            Input:
                - idx: index of item to get
        '''
        return self.files[idx]



class DANRA_Dataset_cutouts_ERA5(Dataset):
    '''
    
    '''
    def __init__(self, 
                data_dir:str,                       # Path to data
                data_size:tuple,                    # Size of data (2D image, tuple)
                n_samples:int=365,                  # Number of samples to load
                cache_size:int=365,                 # Number of samples to cache
                shuffle=False,                      # Whether to shuffle data (or load sequentially)
                cutouts = False,                    # Whether to use cutouts 
                cutout_domains = None,              # Domains to use for cutouts
                n_samples_w_cutouts = None,         # Number of samples to load with cutouts (can be greater than n_samples)
                lsm_full_domain = None,             # Land-sea mask of full domain
                topo_full_domain = None,            # Topography of full domain
                scale=True,                         # Whether to scale data to new interval
                in_low=-1,                          # Lower bound of new interval
                in_high=1,                          # Upper bound of new interval
                data_min_in=-30,                    # Lower bound of data interval
                data_max_in=30,                     # Upper bound of data interval
                data_dir_cond:str=None,             # Path to directory containing conditional data
                conditional=True,                   # Whether to use conditional sampling
                n_classes=4                         # Number of classes for conditional sampling
                ):                       
        '''n_samples_w_cutouts

        '''
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.data_size = data_size
        self.cache_size = cache_size
        self.scale = scale
        self.shuffle = shuffle
        self.cutouts = cutouts
        self.cutout_domains = cutout_domains
        if n_samples_w_cutouts is None:
            self.n_samples_w_cutouts = self.n_samples
        else:
            self.n_samples_w_cutouts = n_samples_w_cutouts
        self.lsm_full_domain = lsm_full_domain
        self.topo_full_domain = topo_full_domain
        self.in_low = in_low
        self.in_high = in_high
        self.data_min_in = data_min_in
        self.data_max_in = data_max_in
        self.conditional = conditional
        self.data_dir_cond = data_dir_cond
        self.n_classes = n_classes

        # Load files from directory (both data and conditional data)
        self.files = sorted(os.listdir(self.data_dir))
        

        # Remove .DS_Store file if present
        if '.DS_Store' in self.files:
            self.files.remove('.DS_Store')
        
        if self.conditional:
            self.files_cond = sorted(os.listdir(self.data_dir_cond))
            if '.DS_Store' in self.files_cond:
                self.files_cond.remove('.DS_Store')

        # Sample n_samples from files, either randomly or sequentially
        if self.cutouts == False:
            if self.shuffle:
                n_samples = min(len(self.files), len(self.files_cond))
                random_idx = random.sample(range(n_samples), n_samples)
                self.files = [self.files[i] for i in random_idx]
                if self.conditional:
                    self.files_cond = [self.files_cond[i] for i in random_idx]
            else:
                self.files = self.files[0:n_samples]
                if self.conditional:
                    self.files_cond = self.files_cond[0:n_samples]
        else:
            if self.shuffle:
                n_samples = min(len(self.files), len(self.files_cond))
                random_idx = random.sample(range(n_samples), self.n_samples_w_cutouts)
                self.files = [self.files[i] for i in random_idx]
                if self.conditional:
                    self.files_cond = [self.files_cond[i] for i in random_idx]
            else:
                n_individual_samples = len(self.files)
                factor = int(np.ceil(self.n_samples_w_cutouts/n_individual_samples))
                self.files = self.files*factor
                self.files = self.files[0:self.n_samples_w_cutouts]
                if self.conditional:
                    self.files_cond = self.files_cond*factor
                    self.files_cond = self.files_cond[0:self.n_samples_w_cutouts]
        
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

        file_path_cond = os.path.join(self.data_dir_cond, self.files_cond[idx])
        file_name_cond = self.files_cond[idx]
        
        if self.conditional:
            if self.n_classes == 4:

                # Determine class from filename
                dateObj = DateFromFile_nc(file_name)
                classifier = dateObj.determine_season()
            
            elif self.n_classes == 12:
                # Determine class from filename
                dateObj = DateFromFile_nc(file_name)
                classifier = dateObj.determine_month()

            elif self.n_classes == 366:
                # Determine class from filename
                dateObj = DateFromFile_nc(file_name)
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
        with nc.Dataset(file_path) as data:
            img = data['t'][0,0,:,:] - 273.15
        
        with nc.Dataset(file_path_cond) as data:
            img_cond = data['t'][0,0,:,:] - 273.15 # ???????????????????????

        if self.cutouts:
            # Get random point
            point = find_rand_points(self.cutout_domains, 128)
            # Crop image, lsm and topo
            img = img[point[0]:point[1], point[2]:point[3]]
            lsm_use = self.lsm_full_domain[point[0]:point[1], point[2]:point[3]]
            topo_use = self.topo_full_domain[point[0]:point[1], point[2]:point[3]]
            if self.conditional:
                img_cond = img_cond[point[0]:point[1], point[2]:point[3]]
        else:
            point = None
        
        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)

            if self.cutouts:
                lsm_use = self.transforms(lsm_use)
                topo_use = self.transforms(topo_use)

            if self.conditional:
                img_cond = self.transforms(img_cond)

        if self.conditional:
            # Return sample image and classifier
            sample = (img, classifier, cond_img)
        else:
            # Return sample image
            sample = (img)
        
        # Add item to cache
        self._addToCache(idx, sample)

        
        if self.cutouts:
            # Return sample image and classifier and random point for cropping (lsm and topo)
            return sample, lsm_use, topo_use, point
        else:
            # Return sample image and classifier only
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
    data_dir_danra_w_cutouts = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_589x789_full/' + var + '_589x789'

    data_dir_lsm = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
    data_dir_topo = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'


    # Set path to save figures
    SAVE_FIGS = False
    PATH_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures'

    # Set number of samples and cache size
    n_samples = 365
    cache_size = 365
    # Set image size
    image_size = (n_danra_size, n_danra_size)
    
    CUTOUTS = True
    CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]

    data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
    data_topo = np.flipud(np.load(data_dir_topo)['data'])
    # Initialize dataset

    #dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size, scale=False, conditional=True, n_classes=12)
    dataset = DANRA_Dataset_cutouts(data_dir_danra_w_cutouts, 
                                    image_size, 
                                    n_samples, 
                                    cache_size, 
                                    cutouts = CUTOUTS, 
                                    shuffle=True, 
                                    cutout_domains = CUTOUT_DOMAINS,
                                    n_samples_w_cutouts = None, 
                                    lsm_full_domain = data_lsm,
                                    topo_full_domain = data_topo,
                                    scale=False, 
                                    conditional=True, 
                                    n_classes=12
                                    )

    # Get sample images
    
    n_samples = 4
    idxs = random.sample(range(0, len(dataset)), n_samples)
    
    # Plot sample image with colorbar
    fig, axs = plt.subplots(1, n_samples, figsize=(15, 4))
    fig2, axs2 = plt.subplots(1, n_samples, figsize=(15, 4))
    fig3, axs3 = plt.subplots(1, n_samples, figsize=(15, 4))

    for idx, ax in enumerate(axs.flatten()):
        (sample_img, sample_season), (sample_lsm, sample_topo, sample_point) = dataset[idx]
        
        axs2.flatten()[idx].imshow(sample_topo)#data_topo[sample_point[0]:sample_point[1], sample_point[2]:sample_point[3]])
        axs2.flatten()[idx].set_title('Topography')
        axs2.flatten()[idx].set_ylim([0, n_danra_size])
        
        axs3.flatten()[idx].imshow(sample_lsm)#data_lsm[sample_point[0]:sample_point[1], sample_point[2]:sample_point[3]])
        axs3.flatten()[idx].set_title('Land-sea mask')
        axs3.flatten()[idx].set_ylim([0, n_danra_size])

        ax.set_ylim([0, n_danra_size])
        
        sample_name = dataset.__name__(idx)

        # Print information about sample image
        print(f'\n\nshape: {sample_img.shape}')
        print(f'season: {sample_season}')
        print(f'min pixel value: {sample_img.min()}')
        print(f'mean pixel value: {sample_img.mean()}')
        print(f'max pixel value: {sample_img.max()}\n')

        #ax.axis('off')
        ax.set_title(sample_name.replace('.npz','') + ', \nclass: ' + str(sample_season))

        img = sample_img.squeeze()
        image = ax.imshow(img, cmap='viridis')
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        
    
    fig.set_tight_layout(True)
    plt.show()



    
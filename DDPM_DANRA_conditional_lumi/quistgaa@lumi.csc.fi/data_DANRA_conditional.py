"""
    Script for generating a pytorch dataset for the DANRA data.
    The dataset is loaded as a single-channel image - either prcp or temp.
    Different transforms are be applied to the dataset.
    A custom transform is used to scale the data to a new interval.
"""
# Import libraries and modules 
import os, random, torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import multiprocessing
import netCDF4 as nc
from multiprocessing import freeze_support
from scipy.ndimage import distance_transform_edt as distance
import zarr

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

def generate_sdf(mask):
    # Ensure mask is boolean
    binary_mask = mask > 0 

    # Distance transform for sea
    dist_transform_sea = distance(~binary_mask)

    # Set land to 1 and subtract sea distances
    sdf = 10*binary_mask.astype(np.float32) - dist_transform_sea

    return sdf

def normalize_sdf(sdf):
    # Find min and max in the SDF
    min_val = np.min(sdf)
    max_val = np.max(sdf)

    # Normalize the SDF
    sdf_normalized = (sdf - min_val) / (max_val - min_val)

    return sdf_normalized

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

class DateFromFile_zarr:
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



class DANRA_Dataset_cutouts_ERA5(Dataset):
    '''
        Class for setting the DANRA dataset with option for random cutouts from specified domains.
        Along with DANRA data, the land-sea mask and topography data is also loaded at same cutout.
        Possibility to sample more than n_samples if cutouts are used.
        Option to shuffle data or load sequentially.
        Option to scale data to new interval.
        Option to use conditional (classifier) sampling (season, month or day).
    '''
    def __init__(self, 
                data_dir:str,                       # Path to data
                data_size:tuple,                    # Size of data (2D image, tuple)
                n_samples:int=365,                  # Number of samples to load
                cache_size:int=365,                 # Number of samples to cache
                variable:str='temp',                 # Variable to load (temp or prcp)
                shuffle=False,                      # Whether to shuffle data (or load sequentially)
                cutouts = False,                    # Whether to use cutouts 
                cutout_domains = None,              # Domains to use for cutouts
                n_samples_w_cutouts = None,         # Number of samples to load with cutouts (can be greater than n_samples)
                lsm_full_domain = None,             # Land-sea mask of full domain
                topo_full_domain = None,            # Topography of full domain
                sdf_weighted_loss = False,          # Whether to use weighted loss for SDF
                scale=True,                         # Whether to scale data to new interval
                in_low=-1,                          # Lower bound of new interval
                in_high=1,                          # Upper bound of new interval
                data_min_in=-30,                    # Lower bound of data interval
                data_max_in=30,                     # Upper bound of data interval
                conditional=True,                   # Whether to use conditional sampling
                data_dir_cond:str=None,             # Path to directory containing conditional data
                n_classes=4                         # Number of classes for conditional sampling
                ):                       
        '''n_samples_w_cutouts

        '''
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.data_size = data_size
        self.cache_size = cache_size
        self.variable = variable
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
        self.sdf_weighted_loss = sdf_weighted_loss
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
            if self.variable == 'temp':
                img = data['t'][0,0,:,:] - 273.15
            elif self.variable == 'prcp':
                img = data['tp'][0,0,:,:] 
        
        with np.load(file_path_cond) as data:
            if self.variable == 'temp':
                img_cond = data['arr_0'][:,:] - 273.15
            elif self.variable == 'prcp':
                img_cond = data['arr_0'][:,:] 
#            [0,0,:,:] - 273.15 # ???????????????????????

        if self.cutouts:
            # Get random point
            point = find_rand_points(self.cutout_domains, 128)
            # Crop image, lsm and topo
            img = img[point[0]:point[1], point[2]:point[3]]
            lsm_use = self.lsm_full_domain[point[0]:point[1], point[2]:point[3]]
            topo_use = self.topo_full_domain[point[0]:point[1], point[2]:point[3]]
            if self.sdf_weighted_loss:
                sdf_use = generate_sdf(lsm_use)
                sdf_use = normalize_sdf(sdf_use)

            if self.conditional:
                img_cond = img_cond[point[0]:point[1], point[2]:point[3]]
        else:
            point = None
        
        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)

            if self.cutouts:
                lsm_use = self.transforms(lsm_use.copy())
                topo_use = self.transforms(topo_use.copy())
                if self.sdf_weighted_loss:
                    sdf_use = self.transforms(sdf_use.copy())

            if self.conditional:
                img_cond = self.transforms(img_cond)

        if self.conditional:
            # Return sample image and classifier
            sample = (img, classifier, img_cond)
        else:   
            # Return sample image
            sample = (img)
        
        # Add item to cache
        self._addToCache(idx, sample)

        
        if self.cutouts:
            if self.sdf_weighted_loss:
                # Return sample image and classifier and random point for cropping (lsm and topo)
                return sample, lsm_use, topo_use, sdf_use, point
            else:
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



class DANRA_Dataset_cutouts_ERA5_Zarr(Dataset):
    '''
        Class for setting the DANRA dataset with option for random cutouts from specified domains.
        Along with DANRA data, the land-sea mask and topography data is also loaded at same cutout.
        Possibility to sample more than n_samples if cutouts are used.
        Option to shuffle data or load sequentially.
        Option to scale data to new interval.
        Option to use conditional (classifier) sampling (season, month or day).
    '''
    def __init__(self, 
                data_dir_zarr:str,                  # Path to data
                data_size:tuple,                    # Size of data (2D image, tuple)
                n_samples:int=365,                  # Number of samples to load
                cache_size:int=365,                 # Number of samples to cache
                variable:str='temp',                # Variable to load (temp or prcp)
                shuffle=False,                      # Whether to shuffle data (or load sequentially)
                cutouts = False,                    # Whether to use cutouts 
                cutout_domains = None,              # Domains to use for cutouts
                n_samples_w_cutouts = None,         # Number of samples to load with cutouts (can be greater than n_samples)
                lsm_full_domain = None,             # Land-sea mask of full domain
                topo_full_domain = None,            # Topography of full domain
                sdf_weighted_loss = False,          # Whether to use weighted loss for SDF
                scale=True,                         # Whether to scale data to new interval
                in_low=-1,                          # Lower bound of new interval
                in_high=1,                          # Upper bound of new interval
                data_min_in=-30,                    # Lower bound of data interval
                data_max_in=30,                     # Upper bound of data interval
                conditional=True,                   # Whether to use conditional sampling
                cond_dir_zarr:str=None,             # Path to directory containing conditional data
                n_classes=4                         # Number of classes for conditional sampling
                ):                       
        '''n_samples_w_cutouts

        '''
        self.data_dir_zarr = data_dir_zarr
        #self.zarr_group_img = zarr_group_img
        self.n_samples = n_samples
        self.data_size = data_size
        self.cache_size = cache_size
        self.variable = variable
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
        self.sdf_weighted_loss = sdf_weighted_loss
        self.in_low = in_low
        self.in_high = in_high
        self.data_min_in = data_min_in
        self.data_max_in = data_max_in
        self.conditional = conditional
        self.cond_dir_zarr = cond_dir_zarr
        #self.zarr_group_cond = zarr_group_cond
        self.n_classes = n_classes

        # Make zarr groups of data
        self.zarr_group_img = zarr.open_group(data_dir_zarr, mode='r')
        self.zarr_group_cond = zarr.open_group(cond_dir_zarr)

        # Load files from directory (both data and conditional data)
        self.files = list(self.zarr_group_img.keys())
        
        if self.conditional:
            self.files_cond = list(self.zarr_group_cond.keys())

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
            Modified to load data from zarr files.
            Input:
                - idx: index of item to get
        '''

        # Get file path, join directory and file name
        # file_path = os.path.join(self.data_dir, self.files[idx])
        file_name = self.files[idx]

        # file_path_cond = os.path.join(self.data_dir_cond, self.files_cond[idx])
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

        # Define zarr groups (img and cond)
        # zarr_group_img = zarr.open_group(self.data_dir_zarr, mode='r')
        # zarr_group_cond = zarr.open_group(self.data_dir_cond_zarr, mode='r')

        # Load data from zarr files, either temp or prcp
        if self.variable == 'temp':
            img = self.zarr_group_img[file_name]['t'][()][0,0,:,:] - 273.15
            img_cond = self.zarr_group_cond[file_name_cond]['arr_0'][()][:,:] - 273.15

        elif self.variable == 'prcp':
            img = self.zarr_group_img[file_name]['tp'][()][0,0,:,:] 
            img_cond = self.zarr_group_cond[file_name_cond]['arr_0'][()][:,:]


        if self.cutouts:
            # Get random point
            point = find_rand_points(self.cutout_domains, 128)
            # Crop image, lsm and topo
            img = img[point[0]:point[1], point[2]:point[3]]
            lsm_use = self.lsm_full_domain[point[0]:point[1], point[2]:point[3]]
            topo_use = self.topo_full_domain[point[0]:point[1], point[2]:point[3]]
            if self.sdf_weighted_loss:
                sdf_use = generate_sdf(lsm_use)
                sdf_use = normalize_sdf(sdf_use)

            if self.conditional:
                img_cond = img_cond[point[0]:point[1], point[2]:point[3]]
        else:
            point = None
        
        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)

            if self.cutouts:
                lsm_use = self.transforms(lsm_use.copy())
                topo_use = self.transforms(topo_use.copy())
                if self.sdf_weighted_loss:
                    sdf_use = self.transforms(sdf_use.copy())

            if self.conditional:
                img_cond = self.transforms(img_cond)

        if self.conditional:
            # Return sample image and classifier
            sample = (img, classifier, img_cond)
        else:   
            # Return sample image
            sample = (img)
        
        # Add item to cache
        self._addToCache(idx, sample)

        
        if self.cutouts:
            if self.sdf_weighted_loss:
                # Return sample image and classifier and random point for cropping (lsm and topo)
                return sample, lsm_use, topo_use, sdf_use, point
            else:
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















# Test the dataset class - .zarr files
if __name__ == '__main__':
    # Use multiprocessing freeze_support() to avoid RuntimeError:
    freeze_support()


    # Set DANRA variable for use
    var = 'prcp'#'temp'#
    # Set size of DANRA images
    n_danra_size = 128#32#64#
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)

    # Set paths to lsm and topo data
    data_dir_lsm = '/scratch/project_465000568/data_lsm/truth_fullDomain/lsm_full.npz'
    data_dir_topo = '/scratch/project_465000568/data_topo/truth_fullDomain/topo_full.npz'

    # Set paths to zarr data
    data_dir_danra_w_cutouts_zarr = '/scratch/project_465000568/data_DANRA/size_589x789_full/zarr_files/' + var + '_589x789_test.zarr'
    data_dir_era5_zarr = '/scratch/project_465000568/data_ERA5/size_589x789/zarr_files/' + var + '_589x789_test.zarr'

    # Set path to save figures
    SAVE_FIGS = False
    PATH_SAVE = '/scratch/project_465000568/DDPM_ouput/Figures'

    # Set number of samples and cache size
    danra_w_cutouts_zarr_group = zarr.open_group(data_dir_danra_w_cutouts_zarr, mode='r')
    n_samples = len(list(danra_w_cutouts_zarr_group.keys()))#365
    cache_size = n_samples//2
    # Set image size
    image_size = (n_danra_size, n_danra_size)
    
    CUTOUTS = True
    CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]

    data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
    data_topo = np.flipud(np.load(data_dir_topo)['data'])

    # Initialize dataset

    #dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size, scale=False, conditional=True, n_classes=12)
    dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_danra_w_cutouts_zarr, 
                                    image_size, 
                                    n_samples, 
                                    cache_size, 
                                    variable = var,
                                    cutouts = CUTOUTS, 
                                    shuffle=True, 
                                    cutout_domains = CUTOUT_DOMAINS,
                                    n_samples_w_cutouts = n_samples, 
                                    lsm_full_domain = data_lsm,
                                    topo_full_domain = data_topo,
                                    sdf_weighted_loss = True,
                                    scale=False, 
                                    conditional=True,
                                    cond_dir_zarr = data_dir_era5_zarr,
                                    n_classes=12
                                    )

    # Get sample images
    
    n_samples = 4
    idxs = random.sample(range(0, len(dataset)), n_samples)
    
    # Plot sample image with colorbar
    fig, axs = plt.subplots(1, n_samples, figsize=(15, 4))
    fig2, axs2 = plt.subplots(1, n_samples, figsize=(15, 4))
    fig3, axs3 = plt.subplots(1, n_samples, figsize=(15, 4))
    fig4, axs4 = plt.subplots(1, n_samples, figsize=(15, 4))
    fig5, axs5 = plt.subplots(1, n_samples, figsize=(15, 4))

    for idx, ax in enumerate(axs.flatten()):
        #(sample_img, sample_season, sample_cond), (sample_lsm, sample_topo, sample_point) = dataset[idx]
        (sample_img, sample_season, sample_cond), sample_lsm, sample_topo, sample_sdf, sample_point = dataset[idx]
        data = dataset[idx]


        axs2.flatten()[idx].imshow(sample_topo[0,:,:])#data_topo[sample_point[0]:sample_point[1], sample_point[2]:sample_point[3]])
        axs2.flatten()[idx].set_title('Topography')
        axs2.flatten()[idx].set_ylim([0, n_danra_size])
        
        axs3.flatten()[idx].imshow(sample_lsm[0,:,:])#data_lsm[sample_point[0]:sample_point[1], sample_point[2]:sample_point[3]])
        axs3.flatten()[idx].set_title('Land-sea mask')
        axs3.flatten()[idx].set_ylim([0, n_danra_size])

        im = axs4.flatten()[idx].imshow(sample_cond[0,:,:], cmap='viridis')#data_lsm[sample_point[0]:sample_point[1], sample_point[2]:sample_point[3]])
        fig4.colorbar(im, ax=axs4.flatten()[idx], fraction=0.046, pad=0.04)
        axs4.flatten()[idx].set_title('ERA5 conditional')
        axs4.flatten()[idx].set_ylim([0, n_danra_size])

        im = axs5.flatten()[idx].imshow(sample_sdf[0,:,:], cmap='viridis')#data_lsm[sample_point[0]:sample_point[1], sample_point[2]:sample_point[3]])
        fig5.colorbar(im, ax=axs5.flatten()[idx], fraction=0.046, pad=0.04)
        axs5.flatten()[idx].set_title('SDF')
        axs5.flatten()[idx].set_ylim([0, n_danra_size])

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
    fig2.set_tight_layout(True)
    fig3.set_tight_layout(True)
    fig4.set_tight_layout(True)
    #plt.show()



    













# # Test the dataset class - .npz files
# if __name__ == '__main__':
#     # Use multiprocessing freeze_support() to avoid RuntimeError:
#     freeze_support()


#     # Set DANRA variable for use
#     var = 'temp'#'prcp'#
#     # Set size of DANRA images
#     n_danra_size = 32#64#128#
#     # Set DANRA size string for use in path
#     danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)

#     # Set paths to data
#     data_dir_danra_w_cutouts = '/scratch/project_465000568/data_DANRA/size_589x789_full/' + var + '_589x789'
#     data_dir_era5 = '/scratch/project_465000568/data_ERA5/size_589x789/' + var + '_589x789'

#     data_dir_lsm = '/scratch/project_465000568/data_lsm/truth_fullDomain/lsm_full.npz'
#     data_dir_topo = '/scratch/project_465000568/data_topo/truth_fullDomain/topo_full.npz'


#     # Set path to save figures
#     SAVE_FIGS = False
#     PATH_SAVE = '/scratch/project_465000568/DDPM_ouput/Figures'

#     # Set number of samples and cache size
#     n_samples = 365
#     cache_size = 365
#     # Set image size
#     image_size = (n_danra_size, n_danra_size)
    
#     CUTOUTS = True
#     CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]

#     data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
#     data_topo = np.flipud(np.load(data_dir_topo)['data'])
#     # Initialize dataset

#     #dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size, scale=False, conditional=True, n_classes=12)
#     dataset = DANRA_Dataset_cutouts_ERA5(data_dir_danra_w_cutouts, 
#                                     image_size, 
#                                     n_samples, 
#                                     cache_size, 
#                                     cutouts = CUTOUTS, 
#                                     shuffle=True, 
#                                     cutout_domains = CUTOUT_DOMAINS,
#                                     n_samples_w_cutouts = None, 
#                                     lsm_full_domain = data_lsm,
#                                     topo_full_domain = data_topo,
#                                     scale=False, 
#                                     conditional=True,
#                                     data_dir_cond = data_dir_era5,
#                                     n_classes=12
#                                     )

#     # Get sample images
    
#     n_samples = 4
#     idxs = random.sample(range(0, len(dataset)), n_samples)
    
#     # Plot sample image with colorbar
#     fig, axs = plt.subplots(1, n_samples, figsize=(15, 4))
#     fig2, axs2 = plt.subplots(1, n_samples, figsize=(15, 4))
#     fig3, axs3 = plt.subplots(1, n_samples, figsize=(15, 4))
#     fig4, axs4 = plt.subplots(1, n_samples, figsize=(15, 4))

#     for idx, ax in enumerate(axs.flatten()):
#         #(sample_img, sample_season, sample_cond), (sample_lsm, sample_topo, sample_point) = dataset[idx]
#         (sample_img, sample_season, sample_cond), sample_lsm, sample_topo, sample_point = dataset[idx]
#         data = dataset[idx]


#         axs2.flatten()[idx].imshow(sample_topo[0,:,:])#data_topo[sample_point[0]:sample_point[1], sample_point[2]:sample_point[3]])
#         axs2.flatten()[idx].set_title('Topography')
#         axs2.flatten()[idx].set_ylim([0, n_danra_size])
        
#         axs3.flatten()[idx].imshow(sample_lsm[0,:,:])#data_lsm[sample_point[0]:sample_point[1], sample_point[2]:sample_point[3]])
#         axs3.flatten()[idx].set_title('Land-sea mask')
#         axs3.flatten()[idx].set_ylim([0, n_danra_size])

#         im = axs4.flatten()[idx].imshow(sample_cond[0,:,:], cmap='viridis')#data_lsm[sample_point[0]:sample_point[1], sample_point[2]:sample_point[3]])
#         fig4.colorbar(im, ax=axs4.flatten()[idx], fraction=0.046, pad=0.04)
#         axs4.flatten()[idx].set_title('ERA5 conditional')
#         axs4.flatten()[idx].set_ylim([0, n_danra_size])

#         ax.set_ylim([0, n_danra_size])
        
#         sample_name = dataset.__name__(idx)

#         # Print information about sample image
#         print(f'\n\nshape: {sample_img.shape}')
#         print(f'season: {sample_season}')
#         print(f'min pixel value: {sample_img.min()}')
#         print(f'mean pixel value: {sample_img.mean()}')
#         print(f'max pixel value: {sample_img.max()}\n')

#         #ax.axis('off')
#         ax.set_title(sample_name.replace('.npz','') + ', \nclass: ' + str(sample_season))

#         img = sample_img.squeeze()
#         image = ax.imshow(img, cmap='viridis')
#         fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        
    
#     fig.set_tight_layout(True)
#     fig2.set_tight_layout(True)
#     fig3.set_tight_layout(True)
#     fig4.set_tight_layout(True)
#     #plt.show()



    
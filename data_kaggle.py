import os, random, tqdm, torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from multiprocessing import Manager as SharedMemoryManager
from multiprocessing import freeze_support

class Scale(object):
    def __init__(self, in_low, in_high, data_min_in, data_max_in):
        self.in_low = in_low
        self.in_high = in_high
        self.data_min_in = data_min_in 
        self.data_max_in = data_max_in

    def __call__(self, sample):
        data = sample
        OldRange = (self.data_max_in - self.data_min_in)
        NewRange = (self.in_high - self.in_low)

        DataNew = (((data - self.data_min_in) * NewRange) / OldRange) + self.in_low
        # DataNew = np.zeros_like(data)
        # for i in range(len(data)):
        #     DataNew[i,:,:] = (((data[i,:,:] - self.data_min_in) * NewRange) / OldRange) + self.in_low

        return DataNew

class DANRA_Dataset(Dataset):
    '''
        Class for setting the DANRA dataset.
        DANRA data is loaded as a single-channel image - either prcp or temp.
        Different transforms can be applied to the dataset.
    '''
    def __init__(self, data_dir:str, data_size:tuple, n_samples:int=365, cache_size:int=365):#, seed=42):
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
        #self.seed = seed

        # Load files from directory
        self.files = sorted(os.listdir(self.data_dir))
        # Remove .DS_Store file if present
        if '.DS_Store' in self.files:
            self.files.remove('.DS_Store')

        # Set seed for reproducibility
        #random.seed(self.seed)
        # Sample n_samples from files
        self.files = random.sample(self.files, self.n_samples)

        self.cache = SharedMemoryManager().dict()
        # Set transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.data_size, antialias=True),
            Scale(-1, 1, -57, 40)#,
            #transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        '''
            Return the length of the dataset.
        '''
        return len(self.files)

    def _addToCache(self, idx:int, data:torch.Tensor):
        if self.cache_size > 0:
            if len(self.cache) >= self.cache_size:
                keys = list(self.cache.keys())
                key_to_remove = random.choice(keys)
                self.cache.pop(key_to_remove)
            self.cache[idx] = data

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

        self._addToCache(idx, img)

        return img
    
    def __name__(self, idx:int):
        '''
            Return the name of the file based on index.
            Input:
                - idx: index of item to get
        '''
        return self.files[idx]




if __name__ == '__main__':
    freeze_support()


    # Set DANRA variable for use
    var = 'temp'#'prcp'#
    # Set size of DANRA images
    n_danra_size = 256#128#
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)

    # Set paths to data
    data_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str

    SAVE_FIGS = False
    PATH_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures'

    epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_channels = 1
    first_fmap_channels = 64
    last_fmap_channels = 512 #2048
    output_channels = 1
    time_embedding = 256
    learning_rate = 1e-4 #1e-2
    min_lr = 1e-6
    weight_decay = 0.0
    n_timesteps = 550
    beta_min = 1e-4
    beta_max = 2e-2
    beta_scheduler = 'linear'
    batch_size = 10
    n_samples = 365
    cache_size = 365
    image_size = (n_danra_size, n_danra_size)
    # seed = random.randint(0, 2**32 - 1)

    print('\n\nTesting data_kaggle.py with multiprocessing freeze_support()\n\n')
    dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size)#, seed=seed)

    sample_img = dataset[0]

    print(f'\n\nshape: {sample_img.shape}')
    print(f'min pixel value: {sample_img.min()}')
    print(f'mean pixel value: {sample_img.mean()}')
    print(f'max pixel value: {sample_img.max()}\n')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title('Sample Image, ' + var)
    img = sample_img.squeeze()#squeeze()#
    
    image = ax.imshow(img, cmap='viridis')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    #plt.imshow(sample_img.permute(1, 2, 0).squeeze(2))
    plt.show()

    
    
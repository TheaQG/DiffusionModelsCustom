


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
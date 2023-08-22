import os
import numpy as np
import torch 
from torch.utils.data import Dataset 
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
#import fnmatch
 

cond_dir = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/era5_multi_channel/era5_4channel/domain_1024x2048/1991'
data_dir = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_128x128/prcp_128x128'

#files = os.listdir(data_dir)
#print(files)





# class era5_dataset(Dataset):
#     def __init__(self, conditional_dir, data_dir, transform=None, conditional_transform=None):
#         self.conditional_dir = conditional_dir
#         self.data_dir = data_dir
#         self.transform = transform
#         self.conditional_transform = conditional_transform

#     def __len__(self):
#         return len(fnmatch.filter(os.listdir(self.data_dir), '*.*'))
    
#     def __getitem__(self, idx):
#         data = np.load(self.data_dir)
#         conditional = np.load(self.conditional_dir)

#         if self.transform:
#             data = self.transform(data)

#         if self.conditional_transform:
#             conditional = self.transform(conditional)

#         return data, conditional
    


# from torch.utils.data import DataLoader

# #train_dataloader = DataLoader()

        


import os
import torch 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# class Resize(object):
#     """
#         Resize the data field to a given size

#         Args:
#             output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
            
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, sample):

#         return
    
# class RandomCrop(object):
# class ToTensor

class Scale(object):
    def __init__(self, data, in_low, in_high, data_min_in, data_max_in):
        self.data = data
        self.in_low = in_low
        self.in_high = in_high
        self.data_min_in = data_min_in 
        self.data_max_in = data_max_in

    def __call__(self, sample):
        OldRange = (self.data_max_in - self.data_min_in)
        NewRange = (self.in_high - self.in_low)

        DataNew = np.zeros_like(self.data)
        for i in range(len(self.data)):
            DataNew[i,:,:] = (((self.data[i,:,:] - self.data_min_in) * NewRange) / OldRange) + self.in_low

        return DataNew
'''
imagen_pytorch.py
-----------------

Contains:
- Imports
- Helper functions
    -  exists: Checks if the given object or value exists (is not None)
    -  identity: Returns the input as is, without any modifications
    -  divisible_by: Checks if a number is divisible by another number
    -  first: Retrieves the first element from an iterable
    -  maybe: Potentially applies a function to a value if the value exists
    -  once: Ensures that a function is called only once
    -  default: Provides a default value in case the given value is None
    -  cast_tuple: Casts the input value to a tuple of the specified length
    -  compact: Removes any None values from a dictionary
    -  maybe_transform_dict_key: Potentially applies a transformation to dictionary keys if the key exists
    -  cast_unit8_images_to_float: Converts images from uint8 to float and normalizes them to the range of 0 to 1
    -  module_device: Retrieves the device of a module (e.g. CPU or GPU) from the first parameter of the module
    -  zero_init_: Initializes the weights of a module to zero. If the module has a bias, the bias is also initialized to zero
    -  eval_decorator: Decorator that sets the module to evaluation mode before calling the function and then restores the module's original mode
    -  pad_tuple_to_length: Pads a tuple to the specified length with the specified fill value

- Helper classes
    - Identity: Identity module that returns the input as is, without any modifications

- Tensor helpers
    - log: Computes the natural logarithm of the input tensor with a small epsilon value added to prevent taking the log of zero
    - l2norm: Computes the L2 norm of the input tensor
    - right_pad_dims_to: Pads the dimensions of the input tensor to match the dimensions of the target tensor
    - masked_mean: Computes the mean of the input tensor along the specified dimension, ignoring masked values
    - resize_image_to: Resizes an image to the specified target size. Resizing is done using bilinear interpolation by default
    - calc_all_frame_dims: Calculates the dimensions of all frames in a video after downsampling
    - safe_get_tuple_index: Retrieves the value at the specified index of a tuple, or returns the default value if the index is out of range

- Image normalization functions
    - normalize_neg_one_to_one: Normalizes an image from the range of 0 to 1 to the range of -1 to 1
    - unnormalize_neg_one_to_one: Unnormalizes an image from the range of -1 to 1 to the range of 0 to 1

- Classifier free guidance functions
    - prob_mask_like: Creates a mask of the specified shape with the specified probability. Used for masking out tokens during training

- Gaussian diffusion with continuous time helper functions and classes
    - beta_linear_log_snr: Computes the log of the signal-to-noise ratio (SNR) using a linear beta schedule
    - alpha_cosine_log_snr: Computes the log of the signal-to-noise ratio (SNR) using a cosine beta schedule



'''
import math # Standard Python module for mathematical operations.
import copy # Standard Python module for shallow and deep copy operations.
from random import random # Standard Python module for generating random numbers.
from beartype.typing import List, Union # Type hints from the beartype library.
from beartype import beartype # Runtime type checking decorator from beartype.
from tqdm.auto import tqdm # tqdm: A library to display progress bars.
from functools import partial, wraps # Higher-order functions and operations on callable objects.
from contextlib import contextmanager, nullcontext # Utilities for working with context management.
from collections import namedtuple # Factory function for creating tuple subclasses with named fields.
from pathlib import Path # Object-oriented filesystem path handling.


import torch # PyTorch: A deep learning framework.
import torch.nn.functional as F # Functional interface for PyTorch's neural network modules.
from torch.nn.parallel import DistributedDataParallel # Utility for parallel and distributed training.
from torch import nn, einsum # Neural network module and Einstein summation convention utility.
from torch.cuda.amp import autocast # Automatic mixed precision training utility.
from torch.special import expm1 # Special mathematical functions from PyTorch. Calculates exp(x) - 1.
import torchvision.transforms as T # Transforms for image and video data. 

import kornia.augmentation as K # Computer vision augmentation library.

from einops import rearrange, repeat, reduce, pack, unpack # Rearrange tensor dimensions.
from einops.layers.torch import Rearrange, Reduce # Rearrange and reduce tensor dimensions.

from t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME # T5 model and tokenizer utilities.

from imagen_video import Unet3D, resize_video_to, scale_video_time # 3D U-Net model and video utilities.

# helper functions

def exists(val):
    '''
    Checks if the given object or value exists (is not None).

    Parameters:
    - val: Any value to check

    Returns:
    - bool: True if value exists, False otherwise
    '''
    return val is not None


def identity(t, *args, **kwargs):
    '''
    Returns the input as is, without any modifications.

    Parameters:
    - t: The input value

    Returns:
    - The unmodified input value
    '''
    return t 


def divisible_by(numer, denom):
    '''
    Checks if a number is divisible by another number.

    Parameters:
    - numer: Numerator to check divisibility
    - denom: Denominator to check divisibility against

    Returns:
    - bool: True if numer is divisible by denom, False otherwise
    '''
    return (numer % denom) == 0


def first(arr, d = None):
    '''
    Retrieves the first element from an iterable.

    Parameters:
    - arr: Iterable from which to retrieve the first element
    - d: Default value to return if iterable is empty

    Returns:
    - The first element of the iterable or the default value if the iterable is empty
    '''
    if len(arr) == 0:
        return d
    return arr[0]


def maybe(fn):
    '''
    Potentially applies a function to a value if the value exists.

    Parameters:
    - fn: ...

    Returns:
    - ...
    '''
    @wraps(fn) # Copies metadata from the original function to the wrapper function.
    def inner(x):
        '''
        Applies the function to the value if the value exists.
        '''
        if not exists(x):
            return x
        return fn(x)
    return inner


def once(fn):
    '''
    Ensures that a function is called only once.

    Parameters:
    - fn: Function to call only once

    Returns:
    - The result of the function call
    '''
    called = False # Boolean flag to track whether the function has been called.
    # Copies metadata from the original function to the wrapper function.
    @wraps(fn)
    def inner(x):
        '''
        Calls the function only once and returns the result.
        '''
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)


def default(val, d):
    '''
    Provides a default value in case the given value is None.

    Parameters:
    - val: Value to check
    - d: Default value or callable returning a default value

    Returns:
    - Value if it exists, otherwise default value
    '''
    return val if exists(val) else d() if callable(d) else d


def cast_tuple(val, length = None):
    '''
    Casts the input value to a tuple of the specified length.
    Casting to a tuple means that if the input value is a list, it will be converted to a tuple.

    Parameters:
    - val: Value to cast to a tuple
    - length: Length of the tuple

    Returns:
    - The input value as a tuple
    '''
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output


def compact(input_dict):
    '''
    Removes any None values from a dictionary.
    This function is used to remove any None values from a dictionary.

    Parameters:
    - input_dict: Dictionary to compact

    Returns:
    - Dictionary with None values removed
    '''
    return {key: value for key, value in input_dict.items() if exists(value)}


def maybe_transform_dict_key(input_dict, key, fn):
    '''
    Potentially applies a transformation to dictionary keys if the key exists.
    Transformations are applied using the provided function.

    Parameters:
    - input_dict: Dictionary to transform
    - key: Key to transform
    - fn: Function to apply to the key

    Returns:
    - Transformed dictionary
    '''
    if key not in input_dict:
        return input_dict
    
    copied_dict = input_dict.copy()
    copied_dict[key] = fn(copied_dict[key])
    return copied_dict


def cast_uint8_images_to_float(images):
    '''
    Converts images from uint8 to float and normalizes them to the range of 0 to 1.

    Parameters:
    - images: Images to convert

    Returns:
    - Converted images
    '''
    if not images.dtype == torch.uint8:
        return images
    
    return images / 255


def module_device(module):
    '''
    Retrieves the device of a module (e.g. CPU or GPU) from the first parameter of the module.    

    Parameters:
    - module: Module from which to retrieve the device (a module is a subclass of the nn.Module class)

    Returns:
    - The device of the module
    '''
    return next(module.parameters()).device


def zero_init_(m):
    '''
    Initializes the weights of a module to zero. If the module has a bias, the bias is also initialized to zero.

    Parameters:
    - m: Module to initialize

    Returns:
    - None
    '''
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


def eval_decorator(fn):
    '''
    Decorator that sets the module to evaluation mode before calling the function and then restores the module's original mode.

    Parameters:
    - fn: Function to call after setting the module to evaluation mode

    Returns:
    - The result of the function call
    '''
    def inner(model, *args, **kwargs):
        was_training = model.training # Boolean flag to track whether the module was in training mode.
        model.eval()
        out = fn(model, *args, **kwargs) # Calls the function.
        model.train(was_training) 
        return out
    return inner


def pad_tuple_to_length(t, length, fill_value = None):
    '''
    Pads a tuple to the specified length with the specified fill value.

    Parameters:
    - t: Tuple to pad
    - length: Length to pad the tuple to
    - fill_value: Value to use for padding, defaults to None

    Returns:
    - Padded tuple
    '''
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fill_value,) * remain_length))

# helper classes

class Identity(nn.Module):
    '''
    Identity module that returns the input as is, without any modifications.
    Not to be confused with the identity function, which is a function that returns its input as is.
    Not sure why this module is needed, but it is used in the Unet class.

    Methods:
    - __init__: Initializes the module
    - forward: Returns the input as is, without any modifications

    '''
    def __init__(self, *args, **kwargs):
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
    
# tensor helpers

def log(t, eps: float = 1e-12):
    '''
    Computes the natural logarithm of the input tensor with a small epsilon value added to prevent taking the log of zero.

    Parameters:
    - t: Tensor to compute the natural logarithm of
    - eps: Small epsilon value to add to the tensor before taking the log, defaults to 1e-12

    Returns:
    - The natural logarithm of the input tensor
    '''
    return torch.log(t.clamp(min = eps))


def l2norm(t):
    '''
    Computes the L2 norm of the input tensor.
    The L2 norm is the square root of the sum of the squares of the tensor's elements.

    Parameters:
    - t: Tensor to compute the L2 norm of

    Returns:
    - The L2 norm of the input tensor 
    '''
    return F.normalize(t, dim = -1)


def right_pad_dims_to(x,t):
    '''
    Pads the dimensions of the input tensor to match the dimensions of the target tensor.

    Parameters:
    - x: Input tensor
    - t: Target tensor

    Returns:
    - The input tensor with padded dimensions
    '''
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t 
    return t.view(*t.shape, *((1,) * padding_dims))


def masked_mean(t, *, dim, mask = None):
    '''
    Computes the mean of the input tensor along the specified dimension, ignoring masked values.

    Parameters:
    - t: Input tensor
    - dim: Dimension along which to compute the mean
    - mask: Mask to apply to the input tensor, defaults to None

    Returns:
    - The mean of the input tensor along the specified dimension, ignoring masked values
    '''
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)


def resize_image_to(image, target_image_size, clamp_range=None, mode='nearest'):
    '''
    Resizes an image to the specified target size. 
    Resizing is done using bilinear interpolation by default.

    Parameters:
    - image: Image to resize
    - target_image_size: Target size to resize the image to
    - clamp_range: Range to clamp the resized image to, defaults to None
    - mode: Mode to use for resizing the image, defaults to 'nearest'

    Returns:
    - The resized image
    '''
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image
    
    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out 


def calc_all_frame_dims(downsample_factors: List[int], frames):
    '''
    Calculates the dimensions of all frames in a video after downsampling.

    Parameters:
    - downsample_factors: List of factors to downsample the frames by
    - frames: Number of frames in the video

    Returns:
    - List of tuples containing the dimensions of all frames in the video after downsampling
    '''
    # If the video has no frames, return a list of empty tuples.
    if not exists(frames):
        return ((tuple(),) * len(downsample_factors))
    

    all_frame_dims = []

    # For each downsample factor, check if the number of frames is divisible by the factor.
    for divisor in downsample_factors:
        assert divisible_by(frames, divisor), f'frames {frames} must be divisible by downsample factor {divisor}'
        all_frame_dims.append((frames // divisor,))

    return all_frame_dims


def safe_get_tuple_index(tup, index, default = None):
    '''
    Retrieves the value at the specified index of a tuple, or returns the default value if the index is out of range.

    Parameters:
    - tup: Tuple to retrieve the value from
    - index: Index of the value to retrieve
    - default: Default value to return if the index is out of range, defaults to None

    Returns:
    - The value at the specified index of the tuple, or the default value if the index is out of range
    '''
    if len(tup) <= index:
        return default
    return tup[index]

# image normalization functions
# ddpms expect images to be in the range of -1 to 1

def normalize_neg_one_to_one(img):
    '''
    Normalizes an image from the range of 0 to 1 to the range of -1 to 1.

    Parameters:
    - img: Image to normalize

    Returns:
    - The normalized image
    '''
    return img * 2. - 1.

def unnormalize_neg_one_to_one(normed_img):
    '''
    Unnormalizes an image from the range of -1 to 1 to the range of 0 to 1.

    Parameters:
    - normed_img: Image to unnormalize

    Returns:
    - The unnormalized image
    '''
    return (normed_img + 1.) / 2.

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    '''
    Creates a mask of the specified shape with the specified probability.
    Used for masking out tokens during training.

    Parameters:
    - shape: Shape of the mask
    - prob: Probability of masking out a token
    - device: Device to create the mask on

    Returns:
    - The mask
    '''
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# gaussian diffusion with continuous time helper functions and classes


@torch.jit.script # Compiles the following function using TorchScript. Improves performance and allows for serialization.
def beta_linear_log_snr(t):
    '''
    Computes the log of the signal-to-noise ratio (SNR) using a linear beta schedule.

    Parameters:
    - t: Input tensor representing the signal values.

    Returns:
    - Log of the signal-to-noise ratio
    '''
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


@torch.jit.script # Compiles the following function using TorchScript. Improves performance and allows for serialization.
def alpha_cosine_log_snr(t, s: float = 0.008):
    '''
    Computes the log of the signal-to-noise ratio (SNR) using a cosine beta schedule 

    Parameters:
    - t: Input tensor representing the signal values.
    - s: Shift value for the cosine function. Default is 0.008.
    
    Returns:
    - Log of the signal-to-noise ratio
    '''
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # not sure if this accounts for beta being clipped to 0.999 in discrete version


def log_snr_to_alpha_sigma(log_snr):
    '''
    Transforms a logarithm of SNR value into two values, alpha and sigma, using the sigmoid function.
    
    Parameters:
    - log_snr: Input tensor representing the logarithm of SNR values.

    Returns:
    - Tuple containing two tensors, alpha and sigma, transformed from the input logarithm of SNR.
    '''
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))



class GaussianDiffusionContinuousTimes(nn.Module):
    '''
    Class to model continuous time Gaussian diffusion processes.

    Methods:
    - __init__: Initializes the class
    - get_times: Returns the times for sampling noise
    - sample_random_times: Samples random times for sampling noise
    - get_condition: Returns the condition for sampling noise
    - get_sampling_timesteps: Returns the sampling timesteps
    - q_posterior: Computes the posterior distribution
    - q_sample: Samples from the posterior distribution
    - q_sample_from_to: Samples from the posterior distribution between two time steps
    '''
    def __init__(self, *, noise_schedule, timesteps = 1000):
        '''
        Initializes the diffusion model with a specific noise schedule ('linear' or 'cosine') and a set number of timesteps.

        Parameters:
        - noise_schedule: Noise schedule to use for the diffusion process
        - timesteps: Number of timesteps to use for the diffusion process
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level, *, device):
        '''
        Creates a tensor of times for sampling noise at a specific noise level.

        Parameters:
        - batch_size: Number of samples in the batch

        Returns:
        - Tensor of times for sampling noise
        '''
        return torch.full((batch_size,), noise_level, device = device, dtype = torch.float32)

    def sample_random_times(self, batch_size, *, device):
        '''
        Samples random times uniformly between 0 and 1 for sampling noise.

        Parameters:
        - batch_size: Number of samples in the batch

        Returns:
        - Tensor of random times for sampling noise
        '''
        return torch.zeros((batch_size,), device = device).float().uniform_(0, 1)

    def get_condition(self, times):
        '''
        Based on the input times, computes the log(SNR) based on the provided noise schedule.
        The function maybe() is used to apply the log(SNR) function to the input times if the times-values exist.
        
        Parameters:
        - times: Times for sampling noise

        Returns:
        - Log of the signal-to-noise ratio (SNR) at the specified times
        '''
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch, *, device):
        '''
        Creates a tensor of sampling timesteps for sampling noise at each timestep.
        The sampling timesteps are the times at which noise is sampled during the diffusion process.
        The 'repeat' function is used to repeat the sampling timesteps for each sample in the batch.
        Unbind is used to split the sampling timesteps into two tensors, one for the start of the timestep and one for the end of the timestep.

        Parameters:
        - batch: Number of samples in the batch
        - device: Device to create the tensor on
        - *: Keyword-only argument

        Returns:
        - Tuple containing two tensors of sampling timesteps, one for the start of the timestep and one for the end of the timestep
        '''
        times = torch.linspace(1., 0., self.num_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    def q_posterior(self, x_start, x_t, t, *, t_next = None):
        '''
        Computes the posterior distribution of the diffusion process given a starting point, x_start, an observation, x_t, and a time, t.
        Equation 33 from the paper is used to compute the posterior distribution.

        Parameters:
        - x_start: Starting point of the diffusion process
        - x_t: Observation of the diffusion process
        - t: Time of the diffusion process

        Returns:
        - Tuple containing three tensors representing the posterior mean, posterior variance, and posterior log variance clipped
        '''

        # If the next time step is not provided, set it to the current time step minus 1 divided by the number of timesteps.
        t_next = default(t_next, lambda: (t - 1. / self.num_timesteps).clamp(min = 0.))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        
        # Compute the log(SNR) at the current time step and the next time step.
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        # Pad the dimensions of the log(SNR) tensors to match the dimensions of the input tensors.
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        # Compute alpha and sigma from the log(SNR) tensors.
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        # following (eq. 33)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        # Prevents the log variance from being too small, clipping it to a minimum value of 1e-20.
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        '''
        Samples from the posterior distribution of the diffusion process given a starting point and a time.

        Parameters:
        - x_start: Starting point of the diffusion process
        - t: Time of the diffusion process
        - noise: Noise to use for sampling, defaults to None

        Returns:
        - Tuple containing four tensors representing the sample, the log of the signal-to-noise ratio (SNR), alpha, and sigma
        '''

        # Set data type of the tensors to the same data type as the starting point.
        dtype = x_start.dtype

        # If the time is a float, set the batch size to the size of the starting point and set the time to a tensor of the specified value.
        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device = x_start.device, dtype = dtype)

        # If the noise is not provided, set it to a tensor of random values with the same shape as the starting point.
        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(t).type(dtype) # Compute the log(SNR) at the specified time.
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr) # Pad the dimensions of the log(SNR) tensor to match the dimensions of the starting point.
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim) # Compute alpha and sigma from the log(SNR) tensor.

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma

    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        '''
        Samples from the posterior distribution of the diffusion process between two times.

        Parameters:
        - x_from: Starting point of the diffusion process
        - from_t: Starting time of the diffusion process
        - to_t: Ending time of the diffusion process
        - noise: Noise to use for sampling, defaults to None

        Returns:
        - Tuple containing four tensors representing the sample, the log of the signal-to-noise ratio (SNR), alpha, and sigma
        
        '''

        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype # Get the shape, device, and data type of the starting point.
        batch = shape[0] # Get the batch size from the shape of the starting point.

        # If the starting time is a float, set the starting time to a tensor of the specified value.
        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device = device, dtype = dtype)

        # If the ending time is a float, set the ending time to a tensor of the specified value.
        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device = device, dtype = dtype)

        # If the noise is not provided, set it to a tensor of random values with the same shape as the starting point.
        noise = default(noise, lambda: torch.randn_like(x_from))

        # Compute the log(SNR) at the starting time and the ending time.
        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to =  log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma * alpha_to) / alpha

    def predict_start_from_v(self, x_t, t, v):
        '''
        Predicts the starting point of the diffusion process given an observation, a time, and a value.

        Parameters:
        - x_t: Observation of the diffusion process
        - t: Time of the diffusion process
        - v: value

        Returns:
        - The predicted starting point of the diffusion process
        
        '''
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return alpha * x_t - sigma * v

    def predict_start_from_noise(self, x_t, t, noise):
        '''
        Predicts the starting point of the diffusion process given an observation, a time, and noise.

        Parameters:
        - x_t: Observation of the diffusion process
        - t: Time of the diffusion process
        - noise: Noise to use for prediction

        Returns:
        - The predicted starting point of the diffusion process
        '''
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-8)
    

# norms and residuals

class LayerNorm(nn.Module):
    '''
    Custom layer normalization module. Layer normalization normalizes across feature dimensions (channels) instead of batch dimensions (batch normalization).

    Methods:
    - __init__: Initializes the module
    - forward: Applies layer normalization to the input tensor
    '''
    def __init__(self, feats, stable = False, dim = -1):
        '''
        Initializes the module with the specified number of features, stability flag, and dimension.

        Parameters:
        - feats: Number of features(neurons)
        - stable: Stability flag, defaults to False. If True, the input tensor is divided by its maximum value before applying layer normalization.
        - dim: Dimension to apply layer normalization across, defaults to -1
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        self.stable = stable
        self.dim = dim

        # Creates a parameter tensor of ones with the specified number of features. It represents the gain (or scale) of the normalization
        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        # If the stability flag is True, divide the input tensor by its maximum value before applying layer normalization.
        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        # A small epsilon value to ensure numerical stability. Its value is determined based on the datatype of x.
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        # Computes the variance and mean of the input tensor across the specified dimension.
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)

# Defines a custom layer normalization module with the dimension set to -3, corresponding to the feature dimension (batch size x channels x height x width).
ChanLayerNorm = partial(LayerNorm, dim = -3)


class Always():
    '''
    A class to create callable objects that always return a fixed value, regardless of the input.
    '''
    def __init__(self, val):
        '''
        Initializes the callable object with the specified value.

        Parameters:
        - val: Fixed value to always return
        '''
        self.val = val

    def __call__(self, *args, **kwargs):
        '''
        Returns the fixed value.
        '''
        return self.val
    

class Residual(nn.Module):
    '''
    Residual/skip connection module, adds the input to the output of the module.
    Is used in for example the ResnetBlock class.

    The line self.fn(x, **kwargs) + x is the essence of a residual connection. 
    Instead of returning just the output of the function/module self.fn, it adds this output back to the original input

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor
    '''
    def __init__(self, fn):
        '''
        Initializes the module with the specified function.
        
        Parameters:
        - fn: Function/neural network module to apply to the input tensor
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        '''
        Applies the module to the input tensor.

        Parameters:
        - x: Input tensor
        '''
        return self.fn(x, **kwargs) + x


class Parallel(nn.Module):
    '''
    Provides a way to run multiple functions or NN modules in parallel on the same input and then sum up their outputs.
    It is a way to combine multiple operations on the same input

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor
    '''
    def __init__(self, *fns):
        '''
        Initializes the module with the specified functions.

        Parameters:
        - *fns: (multiple) Functions to apply to the input tensor

        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        # Creates a list of the functions to apply to the input tensor.
        self.fns = nn.ModuleList(fns) 

    def forward(self, x):
        '''
        Applies the module to the input tensor.
        
        Parameters:
        - x: Input tensor
        '''

        # Applies each function to the input tensor.
        outputs = [fn(x) for fn in self.fns] 
        # Sums up the outputs of the functions and returns.
        return sum(outputs) 
    


# attention pooling


class PerceiverAttention(nn.Module):
    '''
    A module designed to perform multi-head attention, inspired by the Perceiver model but with modifications.

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor

    '''
    def __init__(self, *, dim, dim_head = 64, heads = 8, scale = 8):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor
        - dim_head: Dimensionality of each head, defaults to 64
        - heads: Number of attention heads, defaults to 8
        - scale: Scaling factor, used when computing attention scores, defaults to 8
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads # Computes the total dimensionality for all heads

        self.norm = nn.LayerNorm(dim) # Layer normalization module for input tensor
        self.norm_latents = nn.LayerNorm(dim) # Layer normalization module for latents tensor (the tensor to be attended to)

        self.to_q = nn.Linear(dim, inner_dim, bias = False) # Linear layer to transform the input tensor to the query tensor
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False) # Linear layer to transform the concatenated input tensor and latents to the key and value tensors. Factor 2 because the input tensor and latents are concatenated.

        self.q_scale = nn.Parameter(torch.ones(dim_head)) # Larnable scaling factor for the query tensor
        self.k_scale = nn.Parameter(torch.ones(dim_head)) # Learnable scaling factor for the key tensor

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        ) # Sequential module containing a linear layer and a layer normalization.

    def forward(self, x, latents, mask = None):
        '''
        Applies the module to the input tensor.

        Parameters:
        - x: Main input data
        - latents: The latent representations
        - mask: Optional binary mask to specify which elements should be attended to, defaults to None
        '''
        # Layer normalization for input tensor and latents
        x = self.norm(x)
        latents = self.norm_latents(latents)

        # Get the batch size and number of heads
        b, h = x.shape[0], self.heads

        # Latent representations are transformed to queries
        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        # Input tensor and latents are concatenated and transformed to keys and values
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        # Queries, keys, and values are rearranged to facilitate multi-head attention computations. 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # qk rmsnorm
        # l2 normalization of queries and keys
        q, k = map(l2norm, (q, k))
        # scaling of queries and keys using their respective scaling factors
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities and masking
        # Scaled dot-product between the queries and keys computed to derive attention scores (similarities)
        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale

        # If a mask is provided, apply it to the attention scores
        if exists(mask):
            # positions with False in the mask are set to a very large negative value. 
            # This ensures that these positions don't receive any attention when softmax is applied.
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention
        # Softmax is applied to the attention scores to get the attention weights
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype) # Casts the attention weights to the same data type as the attention scores

        # The attention weights are applied to the values to get the attended values
        out = einsum('... i j, ... j d -> ... i d', attn, v) # Compute weighted sum of values using the attention weights
        out = rearrange(out, 'b h n d -> b n (h d)', h = h) # Rearrange the attended values to match the shape of the input tensor

        return self.to_out(out) # Apply linear layer and layer normalization to the attended values and return the result


class PerceiverResampler(nn.Module):
    '''
    A module designed to perform multi-head attention, inspired by the Perceiver model but with modifications.
    A type of neural network model that processes inputs using a fixed-size set of latent variables, 
    rather than processing the inputs directly. It allows for handling variable-sized inputs with a fixed-sized set of computations.

    In this specific implementation, additional latents are derived from a mean-pooled representation of the input sequence, providing a summarized view of the entire sequence. 
    The combination of individual latents and this summarized view allows the model to capture both detailed and global information from the input.

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor
    '''
    def __init__(self, *, dim, depth, dim_head = 64, heads = 8, num_latents = 64,
        num_latents_mean_pooled = 4, # number of latents derived from mean pooled representation of the sequence
        max_seq_len = 512, ff_mult = 4):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the inputs and latents
        - depth: Number of layers (comprising of attention and feedforward modules)
        - dim_head: Dimensionality of each head, defaults to 64
        - heads: Number of attention heads, defaults to 8
        - num_latents: Number of latent variables, defaults to 64
        - num_latents_mean_pooled: Number of latents derived from a mean-pooled representation of the sequence.
        - max_seq_len: Maximum sequence length, defaults to 512, used for positional embeddings
        - ff_mult: Multiplier for the feedforward network's inner dimension, defaults to 4
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        # Positional embedding for input of size max_seq_len x dim
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Learnable latent variables initialized with random values with the dimensions num_latents x dim
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # Set the latents to None if the number of latents is 0
        self.to_latents_from_mean_pooled_seq = None

        # If greater than 0 
        if num_latents_mean_pooled > 0:
            # A sequential module that transforms the mean-pooled representation of the input sequence to latents.
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled) # Rearrange the tensor to match the shape of the latents
            )

        # Set an empty list of layers
        self.layers = nn.ModuleList([])

        # For each layer, append a PerceiverAttention module (above) and a FeedForward module (below) to the list of layers
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, mask = None):
        '''
        Forward pass of the module.

        Parameters:
        - x: Input tensor
        - mask: Optional binary mask to specify which elements should be attended to, defaults to None
        '''

        # Batch size and device of the input tensor
        n, device = x.shape[1], x.device
        # Positional embedding for input of size n x dim, for positional information
        pos_emb = self.pos_emb(torch.arange(n, device = device))

        # Add positional embedding to input tensor
        x_with_pos = x + pos_emb

        # If the number of latents is greater than 0, repeat the latents for each sample in the batch
        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        # Check if the to_latents_from_mean_pooled_seq module exists
        if exists(self.to_latents_from_mean_pooled_seq):
            # If it exists, input sequence is mean-pooled, transformed and concatenated to the latents
            meanpooled_seq = masked_mean(x, dim = 1, mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim = -2)

        # Input with positional embedding and the latents are passed through the layers
        # Each layer contains:
        # - A Perceiver attention mechanism that updates the latents based on the input and the current latents.
        # - A feed-forward network that updates the latents.
        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask = mask) + latents
            # Latents are updated after each sub-layer with residual connections
            latents = ff(latents) + latents

        return latents



# attention


class Attention(nn.Module):
    '''
    Multi-head attention mechanism, inspired by the Perceiver model but with modifications:
    -  introduces a "null" key-value pair, which can act as a default or fallback attention behavior.
    - It can incorporate an external context, allowing for a kind of two-step attention: first, attending to the context, and then to the main input.
    - It optionally supports relative positional encodings using an attention bias.

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor
    '''
    def __init__(self, dim, *, dim_head = 64, heads = 8, context_dim = None, scale = 8):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor
        - dim_head: Dimensionality of each attention head, defaults to 64
        - heads: Number of attention heads, defaults to 8
        - context_dim: Dimensionality of the context, defaults to None
        - scale: Scaling factor, used when computing attention scores, defaults to 8
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        # Learnable null key-value pair. Initialized with random values with the dimensions 2 x dim_head
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        # Linear layer to transform the input tensor to the query tensor
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # Linear layer to transform the input tensor to the key and value tensors. Factor 2 because the input tensor and latents are concatenated.
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        # Learnable scaling factors for the query and key tensors
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # If a context is provided, this transformation will generate keys and values from the context.
        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        # Linear layer to transform the concatenated input tensor and latents to the output tensor
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context = None, mask = None, attn_bias = None):
        '''
        Forward pass of the module.

        Parameters:
        - x: Main input
        - context: Optional context tensor, defaults to None
        - mask: Optional binary mask to specify which elements should be attended to, defaults to None
        - attn_bias: Optional attention bias tensor, defaults to None, used for relative positional encodings
        '''

        # Batch size, number of heads, and device of the input tensor
        b, n, device = *x.shape[:2], x.device

        # Layer norm of input
        x = self.norm(x)

        # Queries, keys, and values are derived from the input tensor
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        # The queries are rearranged to facilitate multi-head attention computation.
        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        # The null key and value (from self.null_kv) are concatenated to the keys and values.
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # add text conditioning, if present
        if exists(context):
            assert exists(self.to_context)
            # Transform context to keys and values
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            # Concatenate with previous keys and values
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # qk rmsnorm

        # l2 normalization of queries and keys
        q, k = map(l2norm, (q, k))
        # Scaling of queries and keys using their respective scaling factors
        q = q * self.q_scale
        k = k * self.k_scale

        # Calculate query / key similarities (attention scores)
        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # Relative positional encoding (T5 style), if provided
        if exists(attn_bias):
            sim = sim + attn_bias

        # Define a maximum negative value for the attention scores
        max_neg_value = -torch.finfo(sim.dtype).max

        # Mask the attention scores if a mask is provided
        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # Attention weights computed with softmax
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        # Cast the attention weights to the same data type as the attention scores
        attn = attn.to(sim.dtype)

        # Computing sum of values weighted by attention weights
        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # Rearrange the attended values to match the shape of the input tensor
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



# decoder


def Upsample(dim, dim_out = None):
    '''
    Upsamples the input tensor by a factor of 2 using nearest neighbor interpolation, followed by a convolution.

    Parameters:
    - dim: Dimensionality of the input tensor (number of channels)
    - dim_out: Dimensionality of the output tensor, defaults to None, (number of channels)
    '''
    # If the output dimensionality is not provided, set it to the input dimensionality
    dim_out = default(dim_out, dim)

    # Return a sequential module containing an upsampling layer and a convolution layer
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )


class PixelShuffleUpsample(nn.Module):
    """
    PixelShuffle upsample layer, as used in the original StyleGAN paper.
    Pixel Shuffle is known to alleviate checkerboard artifacts.

    PixelShuffle is a method to rearrange elements in a tensor to upscale it, avoiding checkerboard artifacts.
    Rearrangement from shape (b, c * r^2, h, w) shape (b, c, h * r, w * r), where r is the upscaling factor.

    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf

    Methods:
    - __init__: Initializes the module
    - init_conv_: Initializes the convolution layer
    - forward: Applies the module to the input tensor
    """
    def __init__(self, dim, dim_out = None):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor (number of channels)
        - dim_out: Dimensionality of the output tensor, defaults to None, (number of channels)
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        # If output dimensionality not provided, set to input dimensionality
        dim_out = default(dim_out, dim)
        # Convolution layer
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        # Sequential module containing the convolution layer, a SiLU activation function, and a pixel shuffle layer
        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        # Initialize the convolution layer
        self.init_conv_(conv)

    def init_conv_(self, conv):
        '''
        Initializes the convolution layer.

        Parameters:
        - conv: Convolution layer
        '''
        # Get the dimensionality of the convolution layer's weight tensor
        o, i, h, w = conv.weight.shape
        # Initialize the convolution layer's weight tensor with a Kaiming uniform distribution
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        # Repeat the weight tensor 4 times along the first dimension
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        # Copy the weight tensor to the convolution layer's weight tensor
        conv.weight.data.copy_(conv_weight)
        # Initialize the convolution layer's bias tensor to zero
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        '''
        Applies the module to the input tensor.
        '''
        return self.net(x)



def Downsample(dim, dim_out = None):
    '''
    Downsampling method without using pooling or strided convolutions - "pixel unshuffle"
    https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    named SP-conv in the paper, but basically a pixel unshuffle

    The Rearrange operation from the einops library is employed to perform the "pixel unshuffle" operation. 
    In essence, it's the reverse of the pixel shuffle operation.

    Parameters:
    - dim: Dimensionality of the input tensor (number of channels)
    - dim_out: Dimensionality of the output tensor, defaults to None, (number of channels)
    '''
    # If the output dimensionality is not provided, set it to the input dimensionality 
    dim_out = default(dim_out, dim)

    # Return a sequential module containing a convolution layer and a pixel unshuffle layer
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )



class SinusoidalPosEmb(nn.Module):
    '''
    Sinusoidal positional encodings/embedding module.
    Mathematical forms are from the original Transformer paper.

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor
    '''
    def __init__(self, dim):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        self.dim = dim

    def forward(self, x):
        '''
        Applies the module to the input tensor.

        Parameters:
        - x: Input tensor
        '''
        
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1)



class LearnedSinusoidalPosEmb(nn.Module):
    '''
    In this module, the frequency components are learned instead of fixed.
    In standard sinusoidal pos emb, the frequency components are fixed.
    Can give more suitable representations for the sequence positions of the given data.

    following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8


    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor
    '''


    def __init__(self, dim):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        # Make sure the dimensionality is even
        assert (dim % 2) == 0, 'dimensionality of input tensor must be divisible by 2'
        half_dim = dim // 2
        # Learnable weights with the dimensions half_dim (due to the cosine and sine components)
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        '''
        Applies the module to the input tensor.

        Parameters:
        - x: Input tensor
        '''
        # Input tensor is rearranged to match the shape of the weights tensor
        x = rearrange(x, 'b -> b 1')
        # Frequency components are computed by multiplying the input tensor with the weights tensor
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # The frequency components are concatenated to the input tensor after applying sine and cosine functions. Gives sinusoidal encoding for each position
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # Concatenates both the raw positions and the sinusoidal encoding 'ChatGPT says Interesting choice...'
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
class Block(nn.Module):
    '''
    Simple neural network block with normalization, SiLU activation, and a convolution layer.

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor

    '''
    def __init__(self, dim, dim_out, groups = 8, norm = True):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor
        - dim_out: Dimensionality of the output tensor
        - groups: Number of groups for group normalization, defaults to 8
        - norm: Whether to apply normalization, defaults to True

        Attributes:
        - groupnorm: Group normalization module
        - activation: SiLU activation function
        - project: Convolution layer, 3x3 kernel, padding 1
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)

    def forward(self, x, scale_shift = None):
        '''
        Forward pass of the module.

        Parameters:
        - x: Input tensor
        - scale_shift: Optional scale and shift the input tensor, defaults to None

        Computations:
        - Applies group normalization to the input tensor
        - If scale_shift is provided, applies it to the normalized input tensor
        - Applies the activation function to the normalized input tensor
        - Applies the convolution layer to the activated input tensor

        '''
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)

class ResnetBlock(nn.Module):
    '''
    A complex neural network block, inspired by the ResNet architecture - with added features 
    (crossattention, optional time conditioning, squeeze excitation, global context attention).

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor

    '''
    def __init__(self, dim, dim_out, *, cond_dim = None, time_cond_dim = None, groups = 8, linear_attn = False,
                use_gca = False, squeeze_excite = False, **attn_kwargs):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor
        - dim_out: Dimensionality of the output tensor
        - cond_dim: Dimensionality of the context, defaults to None
        - time_cond_dim: Dimensionality of the time embedding, defaults to None
        - groups: Number of groups for group normalization, defaults to 8
        - linear_attn: Whether to use linear attention, defaults to False
        - use_gca: Whether to use global context attention (gca), defaults to False
        - squeeze_excite: Whether to use squeeze excitation, defaults to False
        - attn_kwargs: Keyword arguments for the attention module

        Attributes:
        - time_mlp: Time embedding module (mlp: multi-layer perceptron)
        - cross_attn: Cross attention module if conditional dimensionality is provided
        - block1: First block of the module (from Block class)
        - block2: Second block of the module (from Block class)
        - gca: Global context attention module if use_gca is True
        - res_conv: Convolution layer, 1x1 kernel, padding 0, to match dimensions if input and output dimensionality are different
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        # time embedding
        self.time_mlp = None

        # time conditioning
        if exists(time_cond_dim):
            # Multi-layer perceptron (MLP) module with a SiLU activation function and a linear layer
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        # cross attention
        self.cross_attn = None

        # If conditional dimensionality is provided, cross attention module is initialized
        if exists(cond_dim):
            # If linear attention is True, use LinearCrossAttention module, else use CrossAttention module
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            # Cross attention module
            self.cross_attn = attn_klass(dim = dim_out, context_dim = cond_dim, **attn_kwargs)

        # Block modules
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)

        # Global context attention module
        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)

        # Squeeze excitation module to adaptively recalibrate channel-wise feature responses (matching the input and output dimensions)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()


    def forward(self, x, time_emb = None, cond = None):
        '''
        Forward pass of the module.

        Parameters:
        - x: Input tensor
        - time_emb: Time embedding tensor, defaults to None
        - cond: Context tensor for cross attention, defaults to None

        Computations:
        - If time embedding and time mlp are provided, time embedding is processed and split into scale and shift tensors.
        - The input tensor is passed through the first block
        - If cross attention and context are provided, the input tensor is passed through the cross attention module, unpacked and rearranged to match the shape of the input tensor
        - The input tensor is passed through the second block, with the scale and shift tensors if they exist
        - The input tensor is multiplied by the global context attention module
        - The output is added to the original input after matching the dimensions using the convolution layer
        '''

        # Set scale and shift to None. If time embedding and time mlp are provided, time embedding is processed and split into scale and shift tensors.
        scale_shift = None
        # If time embedding and time mlp are provided 
        if exists(self.time_mlp) and exists(time_emb):
            # Time embedding is processed by the time mlp
            time_emb = self.time_mlp(time_emb)
            # Time embedding is rearranged to match the shape of the input tensor
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # Split the time embedding into scale and shift tensors
            scale_shift = time_emb.chunk(2, dim = 1)

        # Input tensor is passed through the first block
        h = self.block1(x)

        # Check if cross attention exists
        if exists(self.cross_attn):
            # Assert that context exists
            assert exists(cond)
            # Rearrange the input tensor to match the shape of the context tensor
            h = rearrange(h, 'b c h w -> b h w c')
            # The result is reorganized according to batch and channel, with * combining all the dimensions in between, ps is packing structure
            h, ps = pack([h], 'b * c')
            # The result is passed through the cross attention module and added to itself (residual connection)
            h = self.cross_attn(h, context = cond) + h
            # The result is unpacked and rearranged to match the shape of the input tensor
            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b h w c -> b c h w')

        # Result is passed through the second block, with the scale and shift tensors if they exist
        h = self.block2(h, scale_shift = scale_shift)

        # Result is multiplied by the global context attention module
        h = h * self.gca(h)

        # Result is added to the original input after matching the dimensions using the convolution layer
        return h + self.res_conv(x)
    

class CrossAttention(nn.Module):
    '''
    Cross attention module, inspired by the Perceiver model but with modifications.
    Implements a form of multi head attention, where queries come from one source (main input)
    and keys and values come from another (context).
    'Cross' refers to queries/keys/values coming from different sources.
    Uses l2 normalization and cosine similarity for attention scores.

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor

    '''
    def __init__(self, dim, *, context_dim = None, dim_head = 64, heads = 8, norm_context = False, scale = 8):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor
        - context_dim: Dimensionality of the context, defaults to None
        - dim_head: Dimensionality of each attention head, defaults to 64
        - heads: Number of attention heads, defaults to 8
        - norm_context: Whether to apply normalization to the context, defaults to False
        - scale: Scaling factor, used when computing attention scores, defaults to 8

        Attributes:
        - scale: Scaling factor, used when computing attention scores
        - heads: Number of attention heads
        - inner_dim: Inner dim of linear layer, dim_head * heads
        - context_dim: Dimensionality of the context, defaults to dim of input
        - norm: Layer normalization module for the input tensor
        - norm_context: Layer normalization module for the context tensor, defaults to Identity
        - null_kv: Learnable null key-value pair. Initialized with random values with the dimensions 2 x dim_head
        - to_q: Linear layer to transform the input tensor to the query tensor
        - to_kv: Linear layer to transform the context tensor to the key and value tensors. Factor 2 because the input tensor and latents are concatenated.
        - q_scale: Learnable scaling factor for the query tensor
        - k_scale: Learnable scaling factor for the key tensor
        - to_out: Sequential module containing a linear layer and a layer normalization module
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context, mask = None):
        '''
        Forward pass of the module.

        Parameters:
        - x: Main input
        - context: Context tensor
        - mask: Optional binary mask to specify which elements should be attended to, defaults to None
        '''
        # Get batch size, number of heads, and device of the input tensor
        b, n, device = *x.shape[:2], x.device

        # Layer normalization of input and context
        x = self.norm(x)
        context = self.norm_context(context)

        # Queries, keys, and values are derived from the input tensor
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        # Queries, keys, and values are rearranged to facilitate multi-head attention computation.
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # add null key / value for classifier free guidance in prior net
        # The null key and value (from self.null_kv) are concatenated to the keys and values.
        nk, nv = map(lambda t: repeat(t, 'd -> b h 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        # Keys and values are concatenated to the null key and value 
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # cosine sim attention
        # Queries and keys are mapped with l2 normalization and scaled with their respective scaling factors
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities
        # Calculate query / key similarities (attention scores)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # masking
        # Define a maximum negative value for the attention scores
        max_neg_value = -torch.finfo(sim.dtype).max

        # Check if mask exists
        if exists(mask):
            # Pad the mask with a True value
            mask = F.pad(mask, (1, 0), value = True)
            # Rearrange the mask to match the shape of the attention scores
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # Use the mask to mask the attention scores given the maximum negative value
            sim = sim.masked_fill(~mask, max_neg_value)

        # Attention weights computed with softmax
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        # Cast the attention weights to the same data type as the attention scores
        attn = attn.to(sim.dtype)

        # Computing sum of values weighted by attention weights and rearranging the attended values to match the shape of the input tensor
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # The result is passed through the linear layer and layer normalization module
        return self.to_out(out)


class LinearCrossAttention(CrossAttention):
    '''
    A variant of the CrossAttention (above) that uses linear attention mechanisms instead of cosine similarity.
    Linear attention is an approximation to the standard attention mechanism used to reduce computational complexity.

    Methods:
    - forward: Applies the module to the input tensor

    (no need to redefine __init__ because it's the same as CrossAttention)
    Inherit from CrossAttention 
    '''
    def forward(self, x, context, mask = None):
        '''
        Forward pass of the module.

        Parameters:
        - x: Main input (assumed to be object from CrossAttention class)
        - context: Context tensor
        - mask: Optional binary mask to specify which elements should be attended to, defaults to None
        
        '''

        # Get batch size, number of heads, and device of the input tensor
        b, n, device = *x.shape[:2], x.device

        # Layer normalization of input and context
        x = self.norm(x)
        context = self.norm_context(context)

        # Queries, keys, and values are derived from the input tensor
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        # Queries, keys, and values are rearranged to facilitate multi-head attention computation.
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))

        # add null key / value for classifier free guidance in prior net
        # The null key and value (from self.null_kv) are concatenated to the keys and values.
        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # masking
        # Define a maximum negative value for the attention scores
        max_neg_value = -torch.finfo(x.dtype).max

        # Check if mask exists
        if exists(mask):
            # Pad the mask with a True value
            mask = F.pad(mask, (1, 0), value = True)
            # Rearrange the mask to match the shape of the attention scores
            mask = rearrange(mask, 'b n -> b n 1')
            # Use the mask to mask the attention scores given the maximum negative value
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention
        # Queries and keys are activated with a softmax function 
        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        # Queries are scaled
        q = q * self.scale

        # Computing context by multiplying keys and values
        context = einsum('b n d, b n e -> b d e', k, v)
        # Computing final output by multiplying queries and context
        out = einsum('b n d, b d e -> b n e', q, context)
        # Rearrange the attended values to match the shape of the input tensor
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)

        # The result is passed through the linear layer and layer normalization module
        return self.to_out(out)

class LinearAttention(nn.Module):
    '''
    Class for implementing linear attention (no cross attention - i.e. queries, keays and values are derived from the same source).

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor

    '''
    def __init__(self, dim, dim_head = 32, heads = 8, dropout = 0.05, context_dim = None, **kwargs):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor
        - dim_head: Dimensionality of each attention head, defaults to 32
        - heads: Number of attention heads, defaults to 8
        - dropout: Dropout rate, defaults to 0.05
        - context_dim: Dimensionality of the context, defaults to None
        - kwargs: Keyword arguments for the attention module

        Attributes:
        - scale: Square root of the dimensionality of each attention head, used for scaling
        - heads: Number of attention heads
        - inner_dim: Inner dim of linear layer, dim_head * heads
        - norm: Channel-wise layer normalization module for the input tensor
        - nonlin: SiLU activation function (nonlinearity)
        - to_q: Sequential module containing a dropout layer, a convolution layer, and a convolution layer with a 3x3 kernel and padding 1, to transform the input tensor to the query tensor
        - to_k: Sequential module containing a dropout layer, a convolution layer, and a convolution layer with a 3x3 kernel and padding 1, to transform the input tensor to the key tensor
        - to_v: Sequential module containing a dropout layer, a convolution layer, and a convolution layer with a 3x3 kernel and padding 1, to transform the input tensor to the value tensor
        - to_context: Sequential module containing a layer normalization module and a linear layer, to transform the context tensor IF IT EXISTS
        - to_out: Sequential module containing a convolution layer with a 1x1 kernel and a channel-wise layer normalization module

        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context = None):
        '''
        Forward pass of the module.

        Parameters:
        - fmap: Input tensor, feature map from the previous layer
        - context: Context tensor, defaults to None
        '''
        # Get number of heads, and shape of the input feature map
        h, x, y = self.heads, *fmap.shape[-2:]

        # Layer normalization of input fmap 
        fmap = self.norm(fmap)
        # Applies the attribute modules to the input tensor fmap to derive queries, keys, and values
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        # Rearrange the queries, keys, and values to facilitate multi-head attention computation.
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        # If context is provided
        if exists(context):
            # Assert that the to_context attribute exists
            assert exists(self.to_context)
            # Apply the to_context attribute to the context tensor to get context keys and values
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            # Rearrange the context keys and values and concatenate them to the keys and values
            ck, cv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (ck, cv))
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        # Softmax attention 
        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        # Queries are scaled
        q = q * self.scale

        # Computing context by multiplying keys and values
        context = einsum('b n d, b n e -> b d e', k, v)
        # Computing final output by multiplying queries and context
        out = einsum('b n d, b d e -> b n e', q, context)
        # Rearrange the attended values to match the shape of the input tensor
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # The result is passed through the nonlinear activation function and the linear output layer
        out = self.nonlin(out)
        return self.to_out(out)



class GlobalContext(nn.Module):
    """ basically a superior form of squeeze-excitation that is attention-esque """
    '''
    Attention-like mechanism that computes a global context vector for each spatial location in the input tensor.
    Reminiscent of squeeze and excitation (SE) mechanism, but with a more complex attention-like mechanism. (SENets)
    The idea behind squeeze-excitation is to adaptively recalibrate channel-wise feature responses (feature maps) by explicitly modeling interdependencies between channels.

    Computes a set of weights for each spatial location in the input tensor('channel') based on
    This allows the network to emphasize or de-emphasize certain channels based on the overall content of the feature map.

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor

    '''
    def __init__(self, *, dim_in, dim_out):
        '''
        Initializes the module.

        Parameters:
        - dim_in: Dimensionality of the input tensor
        - dim_out: Dimensionality of the output tensor

        Attributes:
        - to_k: Convolution layer with a 1x1 kernel, to transform the input tensor to the key tensor
        - net: Sequential module containing a convolution layer with a 1x1 kernel, a SiLU activation function, a convolution layer with a 1x1 kernel, and a sigmoid activation function

        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        # hidden dim is set to half of output dim
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Forward pass of the module.

        Parameters:
        - x: Input tensor

        '''
        # Compute the global context or 'key' through the convolution layer
        context = self.to_k(x)
        # Map the context and input to a shape suitable for matrix multiplication (flattens spatial dimensions to one)
        x, context = map(lambda t: rearrange(t, 'b n ... -> b n (...)'), (x, context))
        # Computes the weighted sum from input feature map 'x' using weights from softmax of context. Produces a tensor, where each channel is a weighted sum of all spatial positions in the input, based on the context.
        out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
        # Rearrange to add an additional spatial dimension
        out = rearrange(out, '... -> ... 1')
        return self.net(out)


def FeedForward(dim, mult = 2):
    '''
    Simple feedforward module with normalization and SiLU activation.
    Processe flat vectors. 
    Often used as pointwise feedforward module in transformers.

    Inputs:
    - dim: Dimensionality of the input tensor
    - mult: Multiplier for the hidden dimension, defaults to 2
    
    Module layers:
    - LayerNorm: Layer normalization module, normalize input data across feature dimension, mean 0 and var 1
    - Linear: Linear transformation, expanding feature dimension from dim to hidden_dim or hidden_dim to dim
    - GELU: Gaussian Error Linear Unit activation function
    '''
    # Define hidden layer dimension as the input dimension multiplied by the multiplier
    hidden_dim = int(dim * mult)

    # Return a sequential module containing a layer normalization module, a linear layer, and a SiLU activation function
    return nn.Sequential(LayerNorm(dim), nn.Linear(dim, hidden_dim, bias = False), nn.GELU(),
                        LayerNorm(hidden_dim), nn.Linear(hidden_dim, dim, bias = False))

def ChanFeedForward(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    '''
    Feed-forward for 2D data (images/feature maps)
    Especially useful for self-attention layers, where the feedforward is applied to each channel separately.

    Inputs:
    - dim: Dimensionality of the input tensor
    - mult: Multiplier for the hidden dimension, defaults to 2

    Module layers:
    - ChanLayerNorm: Channel-wise layer normalization module
    - Conv2d: Convolution layer, 1x1 kernel, padding 0
    - GELU: Gaussian Error Linear Unit activation function
    - ChanLayerNorm: Channel-wise layer normalization module
    - Conv2d: Convolution layer, 1x1 kernel, padding 0

    '''
    hidden_dim = int(dim * mult)
    return nn.Sequential(ChanLayerNorm(dim), nn.Conv2d(dim, hidden_dim, 1, bias = False), nn.GELU(),
                        ChanLayerNorm(hidden_dim), nn.Conv2d(hidden_dim, dim, 1, bias = False))



class TransformerBlock(nn.Module):
    '''
    Block, with transformer layers. Uses previously defined Attention and FeedForward modules..

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor

    '''
    def __init__(self, dim, *, depth = 1, heads = 8, dim_head = 32, ff_mult = 2, context_dim = None):
        '''
        Initializes the module.
 
        Parameters:
        - dim: Dimensionality of the input tensor
        - depth: Number of transformer layers, defaults to 1
        - heads: Number of attention heads, defaults to 8
        - dim_head: Dimensionality of each attention head, defaults to 32
        - ff_mult: Multiplier for the hidden dimension in the feedforward module, defaults to 2
        - context_dim: Dimensionality of the context if using cross-attention, defaults to None
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        # List to contain the transformer layers
        self.layers = nn.ModuleList([])

        # Append transformer layers to the list for the specified depth
        for _ in range(depth):
            # Append attention and feedforward modules to the list
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        '''
        Forward pass of the module.
        Uses residual connections to ensure that original information is preserved.

        Parameters:
        - x: Input tensor
        - context: Context tensor for cross attention, defaults to None
        '''

        # Rearrange to channels last format
        x = rearrange(x, 'b c h w -> b h w c')
        # The input is packed into a list, and the packing structure is specified
        x, ps = pack([x], 'b * c')

        # Loop through transformer layers (attention and feedforward)
        for attn, ff in self.layers:
            # Apply attention module to the input tensor and add it to itself (residual connection)
            x = attn(x, context = context) + x
            # Apply feedforward module to the input tensor and add it to itself (residual connection)
            x = ff(x) + x

        # Unpack the input tensor and rearrange to channels first format
        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')
        return x
    

class LinearAttentionTransformerBlock(nn.Module):
    '''
    Variation of transformer block with linear attention.
    Allows for faster computation, as complexity is reduced from O(n^2) to O(n).

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor
    '''
    def __init__(self, dim, *, depth = 1, heads = 8, dim_head = 32, ff_mult = 2, context_dim = None, **kwargs):
        '''
        Initializes the module.

        Parameters:
        - dim: Dimensionality of the input tensor
        - depth: Number of transformer layers, defaults to 1
        - heads: Number of attention heads, defaults to 8
        - dim_head: Dimensionality of each attention head, defaults to 32
        - ff_mult: Multiplier for the hidden dimension in the feedforward module, defaults to 2
        - context_dim: Dimensionality of the context if using cross-attention, defaults to None
        - kwargs: Keyword arguments for the attention module
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        # List to contain the transformer layers
        self.layers = nn.ModuleList([])

        # Loop through transformer layers (attention and feedforward)
        for _ in range(depth):
            # Append attention and feedforward modules to the list
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),
                ChanFeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x, context = None):
        '''
        Forward pass of the module.

        Parameters:
        - x: Input tensor
        - context: Context tensor for cross attention, defaults to None
        '''

        # Loop through transformer layers (attention and feedforward)
        for attn, ff in self.layers:
            # Apply attention module to the input tensor and add it to itself (residual connection)
            x = attn(x, context = context) + x
            # Apply feedforward module to the input tensor and add it to itself (residual connection)
            x = ff(x) + x
        return x
    

class CrossEmbedLayer(nn.Module):
    '''
    Applies mutliple convolutions with different kernel sizes to the input.

    The class is designed to capture multiscale features from an input tensor by applying convolutions 
    with different kernel sizes. The concatenated results provide a richer representation of the input, 
    with features extracted at various scales.

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor
    '''
    def __init__(self, dim_in, kernel_sizes, dim_out = None, stride = 2):
        '''
        Initializes the module.

        Parameters:
        - dim_in: Dimensionality of the input tensor
        - kernel_sizes: List of kernel sizes for the convolutions
        - dim_out: Dimensionality of the output tensor, defaults to None
        - stride: Stride of the convolutions, defaults to 2
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        # Check if kernel sizes and stride are either both even or both odd to ensure compatible padding values
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])

        # Set output dimensionality to input dimensionality if not specified
        dim_out = default(dim_out, dim_in)

        # Sort the kernel sizes in ascending order
        kernel_sizes = sorted(kernel_sizes)
        # Get the number of kernel sizes
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]

        # Calculate the dimension at the last scale
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        # Create a list of convolution layers with the specified kernel sizes and dimensionality
        self.convs = nn.ModuleList([])
        # Loop through the kernel sizes and dimensionality
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            # Append a convolution layer to the list for each kernel size and dimensionality
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        '''
        Forward pass of the module.
        '''
        # Apply each convolution layer to the input tensor and concatenate the results
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)
    

class UpsampleCombiner(nn.Module):
    '''
    Handles upsampling and combining of feature maps. Used to integrate information from different scales or resolutions.

    Methods:
    - __init__: Initializes the module
    - forward: Applies the module to the input tensor
    '''
    def __init__(self, dim, *, enabled = False, dim_ins = tuple(), dim_outs = tuple()):
        '''
        Initializes the module.
        
        If 'enabled' is True, for each feature map dimension in dim_ins, the module initializes a block (Block) 
        that transforms it to its corresponding dimension in dim_outs. These blocks are stored in self.fmap_convs.

        Parameters:
        - dim: Input feature dimension (number of channels)
        - enabled: Whether to enable the module, defaults to False (acts as identity module, pass-through)
        - dim_ins: Tuple representing the input dimensions of each of the feature maps
        - dim_outs: Dimensionality of the output tensors after transformations 
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()
        # Cast dim_ins and dim_outs to tuples
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        # Check if the number of input dimensions and output dimensions are the same
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        # Check if the module is enabled, if not, set the output dimensionality to the input dimensionality
        if not self.enabled:
            self.dim_out = dim
            return

        # Use the Block class to create a list of blocks that transform the input feature maps to the specified output dimensions
        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        # Set the output dimensionality to the input dimensionality plus the sum of the output dimensions of the blocks
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps = None):
        '''
        Forward pass of the module.
        If enabled is False or if there are no feature maps provided or no transformation blocks initialized, the input x is returned as-is.

        Parameters:
        - x: Input tensor
        - fmaps: Optional list of feature maps to be transformed, defaults to None
        '''
        # Define the target size as the size of the last dimension of the input tensor
        target_size = x.shape[-1]

        # Set the feature maps to an empty tuple if not provided
        fmaps = default(fmaps, tuple())

        # Check if the module is enabled, if there are no feature maps provided, or if there are no transformation blocks initialized
        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            # If so, return the input tensor as-is
            return x

        # Resize the feature maps to match the target sizes
        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        # Apply each transformation block to the corresponding feature map
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        # Concatenate the input tensor with the transformed feature maps and return the result
        return torch.cat((x, *outs), dim = 1)



class Unet(nn.Module):
    def __init__(
        self,
        *,
        dim,                                                # base dimension of the model (number of neurons in the first layer and used as defaults)
        text_embed_dim = get_encoded_dim(DEFAULT_T5_NAME),  # dimensionality of the text embeddings, defaults to the encoded dimensionality of the T5 model
        num_resnet_blocks = 1,                              # Number of resnet blocks, defaults to 1. Single integer (applied uniformly to all layers) or tuple specifying number for each layer
        cond_dim = None,                                    # Dimensionality of the conditioning tensor, defaults to None
        num_image_tokens = 4,                               # Number of tokens used for image embedding when model is conditioned on images
        num_time_tokens = 2,                                # Number of tokens used for time embeddings 
        learned_sinu_pos_emb_dim = 16,                      # Dimensionality of the learned sine positional embedding, defaults to 16
        out_dim = None,                                     # Dimensionality of the U-Net's output tensor, defaults to None
        dim_mults=(1, 2, 4, 8),                             # Dimensionality multipliers for each layer, defaults to (1, 2, 4, 8), e.g. 64 --> 128 --> 256 --> 512
        cond_images_channels = 0,                           # Number of channels in the conditioning images, defaults to 0
        channels = 3,                                       # Number of channels in the input image, defaults to 3
        channels_out = None,                                # Number of channels in the output image, defaults to None
        attn_dim_head = 64,                                 # Dimensionality of each attention head, defaults to 64
        attn_heads = 8,                                     # Number of attention heads, defaults to 8
        ff_mult = 2.,                                       # Multiplier for the feed-forward layer dimensions in the attention mechanisms, defaults to 2
        lowres_cond = False,                                # Whether to use low resolution conditioning, for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns = True,                                 # Whether to use self-attention blocks at each layer, defaults to True
        layer_attns_depth = 1,                              # Number of layers (depth) of self-attention blocks, defaults to 1
        layer_mid_attns_depth = 1,                          # Number of layers (depth) of self-attention blocks at the bottleneck, defaults to 1
        layer_attns_add_text_cond = True,                   # Whether to condition the self-attention blocks with the text embeddings, as described in Appendix D.3.1
        attend_at_middle = True,                            # Whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        layer_cross_attns = True,                           # Flag, or tuple of flags specifying whether to use cross-attention blocks at each layer, defaults to True
        use_linear_attn = False,                            # Whether to use linear attention instead of full attention, defaults to False
        use_linear_cross_attn = False,                      # Whether to use linear attention instead of full attention for cross-attention, defaults to False
        cond_on_text = True,                                # Flag indicating whether to condition on text, defaults to True
        max_text_len = 256,                                 # Maximum length of the text, defaults to 256
        init_dim = None,                                    # Dimensionality of the initial convolutional layer, defaults to None
        resnet_groups = 8,                                  # Number of groups in the ResNet blocks' group convolutions, defaults to 8
        init_conv_kernel_size = 7,                          # kernel size of initial convolution, if not using cross embed
        init_cross_embed = True,                            # Whether to use cross embedding for the initial convolution, defaults to True
        init_cross_embed_kernel_sizes = (3, 7, 15),         # Kernel sizes for the cross embedding, defaults to (3, 7, 15)
        cross_embed_downsample = False,                     # Whether to use cross embedding for downsampling, defaults to False
        cross_embed_downsample_kernel_sizes = (2, 4),       # Tuple of kernel sizes for the cross embedding during downsampling, defaults to (2, 4)
        attn_pool_text = True,                              # Whether to use attention pooling for text embeddings, defaults to True
        attn_pool_num_latents = 32,                         # Number of latents to use for attention pooling, defaults to 32
        dropout = 0.,                                       # Dropout rate, defaults to 0
        memory_efficient = False,                           # Whether to use checkpointing to reduce memory usage, defaults to False
        init_conv_to_final_conv_residual = False,           # Whether to use a residual connection from the initial convolution to the final convolution, defaults to False
        use_global_context_attn = True,                     # Whether to use global context attention, defaults to True
        scale_skip_connection = True,                       # Whether to scale the skip connection, defaults to True
        final_resnet_block = True,                          # Whether to use a resnet block before final convolution, defaults to True
        final_conv_kernel_size = 3,                         # Kernel size of the final convolution, defaults to 3
        self_cond = False,                                  # Whether to condition on the predicted image, defaults to False
        resize_mode = 'nearest',                            # What resize mode to use for the upsampling, defaults to 'nearest' - 'nearest' or 'bilinear'
        combine_upsample_fmaps = False,                     # Combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = True,                      # May address checkboard artifacts
    ):
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        # If the number of attention heads is less than 1, raise an error (at least 1 attention head is required)
        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        # Check if the base dimension is less than 128, and recommend the user to increase it if so
        if dim < 128:
            print_once('The base dimension of your u-net should ideally be no smaller than 128, as recommended by a professional DDPM trainer https://nonint.com/2022/05/04/friends-dont-let-friends-train-small-diffusion-models/')

        # Save local variables as attributes, removing self and __class__ from it 
        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        
        # BASIC MODEL DIMENSIONS #

        self.channels = channels
        # Set the output dimensionality to the input dimensionality if not specified
        self.channels_out = default(channels_out, channels)

        # (1) in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        # (2) in self conditioning, one appends the predict x0 (x_start)

        # Add channels if using low resolution conditioning(+(int(N_lowres_cond))) and self conditioning (+int(N_self_cond))
        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))
        # Set the initial dimensionality to the input dimensionality if not specified
        init_dim = default(init_dim, dim)

        self.self_cond = self_cond


        # OPTIONAL IMAGE CONDITIONING #

        # Check if number of channels from conditioning image is greater than 0 (i.e. it exists)
        self.has_cond_image = cond_images_channels > 0
        self.cond_images_channels = cond_images_channels

        # Add number of channels from low ress if using low resolution conditioning
        init_channels += cond_images_channels


        # INITIAL CONVOLUTION #

        # Initial convolution with either cross embedding or normal convolution
        if init_cross_embed:
            self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1) 
        else:
            nn.Conv2d(init_channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

        # Define the dimensionality resulting from upsampling, which is the base dimension multiplied by the dimensionality multipliers
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # Make a list containing tuples of the dimensionality of the input and output tensors for each layer (pairs the dimensions to represent in- and output dimensions of each layer)
        in_out = list(zip(dims[:-1], dims[1:]))


        # TIME CONDITIONING #

        # Set the dimensionality of the conditioning tensor to the base dimension if not specified
        cond_dim = default(cond_dim, dim)
        # Set the dimensionality of the time conditioning tensor to 4 * base dimension * 2/1 (lowres_cond/no lowres_cond)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        # Setting up a learned sinusoidal positional embedding for time conditioning. Sinusoidal positional embedding time for log(snr) noise from continuous version
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        # Add 1 to the dimensionality of the sinusoidal positional embedding to account for the time token
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1


        # Setting up a model to process time embeddings, hidden layers
        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )

        # Set up a model to determine the time conditioning
        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # Project the time conditioning to multiple time tokens
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )


        # LOW RESOLUTION CONDITIONING #
        # low res aug noise conditioning
        
        self.lowres_cond = lowres_cond

        # If there a low resolution condition is available, we create models (as for the time conditioing) to process the low resolution conditioning
        if lowres_cond:
            # Set a model to process low resolution time conditioning (hidden layers)
            self.to_lowres_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim),
                nn.Linear(learned_sinu_pos_emb_dim + 1, time_cond_dim),
                nn.SiLU()
            )

            # Set a model to determine the low resolution time conditioning
            self.to_lowres_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )

            # Make a model to project the low resolution time conditioning to multiple time tokens (including the low-res condition in the time conditioning, i.e. project them to the same tokens)
            self.to_lowres_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange('b (r d) -> b r d', r = num_time_tokens)
            )

        # Layer normalization for the conditioning dimension
        self.norm_cond = nn.LayerNorm(cond_dim)

        # TEXT ENCODING CONDITIONING #
        # Optional conditioning based on text embeddings

        self.text_to_cond = None

        # If the model is conditioned on text
        if cond_on_text:
            # Make sure that the text embedding dimensionality is specified
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text is True'
            # Make a single layer to project the text embeddings to the desired dimension (the conditioning dimensionality)
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # Finer control over whether to condition on text encodings at all points in the network where time is also conditioned
        # If the model is conditioned on text
        self.cond_on_text = cond_on_text

        # Attention mechanism to potentially pool over text embeddings (perceiver resampler, i.e. attention pooling)
        self.attn_pool = PerceiverResampler(dim = cond_dim, depth = 2, dim_head = attn_dim_head, heads = attn_heads, num_latents = attn_pool_num_latents) if attn_pool_text else None

        # For classifier free guidance
        self.max_text_len = max_text_len

        # Set a null text embedding and hidden to be used if no text is provided
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

        # For non-attention based text conditioning at all points in the network where time is also conditioned
        self.to_text_non_attn_cond = None

        # If the model is conditioned on text
        if cond_on_text:
            # Set the text conditioning with no attention
            self.to_text_non_attn_cond = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim)
            )

        # Make a dictionary of keyword arguments for the attention mechanism (number of heads, dimensionality of each head)
        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)
        # Set the number of layers to the length of the dimensionality of input/output tuples
        num_layers = len(in_out)

        
        # RESNET BLOCKS CONFIGURATION #
        
        # Cast the number of resnet blocks to a tuple if it is not already one
        # (cast_tuple ensures that the inputs are expanded or truncated to match the number of layers in the U-Net)
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        # Cast the number of resnet groups to a tuple if it is not already one
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        # Create a partially initialized instance of ResnetBlock with the previously defined keyword arguments
        # (Partially initialized to allow for specification of other arguments later)
        resnet_klass = partial(ResnetBlock, **attn_kwargs)

        # Cast the layer attention flags, depths and attentions to tuples if not already one
        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        # Cast the linear attention flags and cross attentions to tuples if not already one
        use_linear_attn = cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, num_layers)

        # Check if the number of resnet blocks, layer attentions, layer cross attentions, and linear attentions are all the same
        assert all([layers == num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])


        # DOWNSAMPLING CONFIGURATION #

        # Define a downsample instance 
        downsample_klass = Downsample

        # If cross embedding is used for downsampling
        if cross_embed_downsample:
            # Set the downsample instance to be a cross embedding layer
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)


        # Initial ResNet block

        # Set up initial ResNet block. Used for memory efficient unet. If not, block is set to None.  
        self.init_resnet_block = resnet_klass(init_dim, init_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[0], use_gca = use_global_context_attn) if memory_efficient else None

        # Scale for resnet skip connections, defaults to 1 if not specified, otherwise 2^(-1/2)
        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # Initialize empty module lists for up- and downsampling layers of U-Net
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        # Define the number of resolutions (i.e. the number of layers in the U-Net)
        num_resolutions = len(in_out)

        # Define a list containing the parameters for each part of the network
        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_attns_depth, layer_cross_attns, use_linear_attn, use_linear_cross_attn]
        # Reverse the list of parameters
        reversed_layer_params = list(map(reversed, layer_params))


        # downsampling layers

        # Define a list to keep track of the skip connection dimensions
        skip_connect_dims = [] 

        # Loop to construct downsampling part of the U-Net. For each layer if defines the relevant operations
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            # Check if the current index is the last one
            is_last = ind >= (num_resolutions - 1)

            # Set the dimensionality of the conditioning tensor to the base dimension if using cross attention of any kind. Otherwise none
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            # Check if layer attention is used
            if layer_attn:
                # If regular attention is used, set the transformer block class to the regular transformer block
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                # If linear attention is used, set the transformer block class to the linear attention transformer block
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                # If no attention is used, set the transformer block class to the identity function (no attention)
                transformer_block_klass = Identity

            # Set the current dimensionality to the input dimensionality
            current_dim = dim_in

            # Whether to pre-downsample, from memory efficient unet
            pre_downsample = None

            # If memory efficient unet is used, the downsampling is performed before the resnet block
            if memory_efficient:
                # Set the pre-downsample to be a downsample object
                pre_downsample = downsample_klass(dim_in, dim_out)
                # Set the current dimensionality to the output dimensionality of the downsampling
                current_dim = dim_out

            # Append the current dimensionality to the list of skip connection dimensions
            skip_connect_dims.append(current_dim)

            # Whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out) if not is_last else Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), nn.Conv2d(dim_in, dim_out, 1))

            # Append the downsampling layers to the list of downsampling layers, with pre_/post_downsample either None or the downsample object
            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, current_dim, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(current_dim, current_dim, time_cond_dim = time_cond_dim, groups = groups, use_gca = use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
                post_downsample
            ]))


        # MIDDLE LAYERS #
        # The layers between downsampling and upsampling

        # Set the dimensionality of the middle layers to the last dimensionality of the input/output tuples (i.e. the dimensionality of the bottleneck)
        mid_dim = dims[-1]

        # Define a ResNet-block, an (optional) attention block, and another ResNet-block for the middle layers
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])
        self.mid_attn = TransformerBlock(mid_dim, depth = layer_mid_attns_depth, **attn_kwargs) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])


        # UPSAMPLING CONFIGURATION #

        # Define what upsample class to use, either regular or pixel shuffle
        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # Initialize an empty list to keep track of the feature map dimensions during upsampling
        upsample_fmap_dims = []

        # Main loop to construct the upsampling part of the U-Net. For each layer it defines the relevant operations. Same as downsampling loop, but reversed.
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            # Check if the current index is the last one
            is_last = ind == (len(in_out) - 1)

            # Set the dimensionality of the conditioning tensor to the base dimension if using cross attention of any kind. Otherwise none
            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            # Check if layer attention is used
            if layer_attn:
                # If regular attention is used, set the transformer block class to the regular transformer block
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                # If linear attention is used, set the transformer block class to the linear attention transformer block
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                # If no attention is used, set the transformer block class to the identity function (no attention)
                transformer_block_klass = Identity

            # Define the skip-connect dimension as the last element of the list of skip connection dimensions
            skip_connect_dim = skip_connect_dims.pop()

            # Add the output dimensionality to the list of feature map dimensions during upsampling
            upsample_fmap_dims.append(dim_out)

            # Append the upsampling layers to the module list, remembering skip connection dimensions and going reverse order of the downsampling layers
            self.ups.append(nn.ModuleList([
                resnet_klass(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, linear_attn = layer_use_linear_cross_attn, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(dim_out + skip_connect_dim, dim_out, time_cond_dim = time_cond_dim, groups = groups, use_gca = use_global_context_attn) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = dim_out, depth = layer_attn_depth, ff_mult = ff_mult, context_dim = cond_dim, **attn_kwargs),
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))


        # FINAL LAYERS #

        # Operation to combine the feature maps from all upsampling blocks before the final resnet block out
        self.upsample_combiner = UpsampleCombiner(
            dim = dim,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = dim
        )

        # Whether to make residual connection from initial convolution to final ResNet block output
        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        # Set the dimensionality of the final convolution to the dimensionality of the upsampling combiner  - plus the dimensionality of the initial convolution if using residual connection
        final_conv_dim = self.upsample_combiner.dim_out + (dim if init_conv_to_final_conv_residual else 0)

        # Defines a final (optional) ResNet block
        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim = time_cond_dim, groups = resnet_groups[0], use_gca = True) if final_resnet_block else None

        # Defines the final convolution dimensionality to be either the final ResNet block dimensionality or the final convolution dimensionality
        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        # Add the number of channels from the low resolution conditioning if using low resolution conditioning
        final_conv_dim_in += (channels if lowres_cond else 0)

        # Define the final convolution to produce the output of the U-Net
        self.final_conv = nn.Conv2d(final_conv_dim_in, self.channels_out, final_conv_kernel_size, padding = final_conv_kernel_size // 2)

        # Initialize the weights of the final convolution to zero (ensuring stable training)
        zero_init_(self.final_conv)

        # Store the resize mode for resizing operations. Either 'nearest' or 'bilinear'
        self.resize_mode = resize_mode

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(self, *, lowres_cond, text_embed_dim, channels, channels_out, cond_on_text):
        '''
        Checks if certain parameters of the current model match a given set of parameters. If they match, the model instance
        is simply returned. If not, an updated set of keyword arguments is prepared with the new parameters, and a new model
        instance is returned with the updated parameters.
        The * operator in the function definition is used to force the user to specify the keyword arguments by name, and not
        by position.

        Parameters:
        - lowres_cond (bool): Whether to use low resolution conditioning
        - text_embed_dim (int): Dimensionality of the text embeddings
        - channels (int): Number of channels in the input image
        - channels_out (int): Number of channels in the output image
        - cond_on_text (bool): Whether to condition on text or not

        Returns:
        - self (Unet): The current model instance, or a new model instance with the updated parameters
        '''
        # Checking if all the parameters match the current model
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            channels_out == self.channels_out:
            # If they all match, return the current model
            return self

        # If they do not match, prepare an updated set of keyword arguments with the new parameters
        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text
        )

        # Return a new model instance with the updated parameters. The ** operator in the function call is used to unpack the
        # keyword arguments from the dictionary into the function call. _locals is a dictionary containing all the local variables, 
        # including the keyword arguments, of the current model instance. __class__ is a local variable that contains the class of
        # the current model instance. 
        # **{**self._locals, **updated_kwargs} unpacks the local variables into a dictionary, and then
        # unpacks the updated keyword arguments into the same dictionary, overwriting the old values with the new ones.
        return self.__class__(**{**self._locals, **updated_kwargs})


    def to_config_and_state_dict(self):
        '''
        Returns two pieces of information about the current model instance: the configuration, and the state dictionary.
        Thus information on both the state AND architecture of the model is available.

        Returns:
        - self._locals: A saved dictionary of the model's configuration
        - self.state_dict(): Dictionary containing the entire state of the module(model) - all the model's parameters like weights and biases
        '''
        return self._locals, self.state_dict()
    

    # @classmethod is different from ordinary methods in the class, as it has access to the class itself and not just the 
    # instance of the class, through the cls argument (here 'klass').
    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        '''
        Creates a new model instance from a saved configuration and state dictionary. This is useful for loading a model
        from a saved file and continuing training on it with the latest settings.

        Parameters:
        - klass (Unet): Reference to the class itself
        - config (dict): The saved configuration of the model
        - state_dict (dict): The saved state dictionary of the model

        Returns:
        - unet (Unet): A new model instance with the saved configuration and state dictionary
        '''
        # Create a new model instance with the saved configuration
        unet = klass(**config)
        # Load the saved state dictionary into the new model instance
        unet.load_state_dict(state_dict)
        # Return the new model instance
        return unet


    def persist_to_file(self, path):
        '''
        Save both architecture (configuration) and learned parameters (state dictionary) of the current model instance to a file.

        Parameters:
        - path (str): The path to the file to save the model to
        '''
        # Convert path to a Path object
        path = Path(path)
        # Check if the path exists, and if not, create it, including any parent directories
        path.parents[0].mkdir(exist_ok = True, parents = True)

        # Get the configuration and state dictionary of the current model instance
        config, state_dict = self.to_config_and_state_dict()
        # Save the configuration and state to a dictionary
        pkg = dict(config = config, state_dict = state_dict)
        # Save the dictionary to the specified path
        torch.save(pkg, str(path))


    @classmethod
    def hydrate_from_file(klass, path):
        '''
        Method to load a model's architecture and learned parameters from a file and return an instance
        of the model with the loaded parameters.

        Parameters:
        - klass (Unet): Reference to the class itself
        - path (str): The path to the file to load the model from

        Returns:
        - unet (Unet): A new model instance with the loaded configuration and state dictionary
        '''
        # Convert path to a Path object
        path = Path(path)
        # Make sure the path exists
        assert path.exists()
        # Load the saved dictionary from the specified path. (expected to be the dict saved using persist_to_file)
        pkg = torch.load(str(path))

        # Make sure the saved dictionary contains both the configuration and state dictionary
        assert 'config' in pkg and 'state_dict' in pkg
        # Get the configuration and state dictionary from the saved dictionary
        config, state_dict = pkg['config'], pkg['state_dict']

        # Create a new model instance with the loaded configuration and return it
        return Unet.from_config_and_state_dict(config, state_dict)


    def forward_with_cond_scale(self, *args, cond_scale = 1., **kwargs):
        '''
        Forward with classifier free guidance.

        Method to allow for controlled blending between the regular model output (logits) and a version of the output where some 
        conditioning is dropped (null_logits). The degree of blending is controlled by cond_scale. 
        This can be useful in scenarios where you want to analyze the influence of certain conditioning on the model's output or gradually 
        transition between two different modes of operation.
        
        Parameters:
        - args: Variable length argument list
        - cond_scale (float): The degree of blending between the regular model output and the null_logits.
        - kwargs: Arbitrary keyword arguments
        '''

        # Call the 'forward' method (below) and store output in 'logits'
        logits = self.forward(*args, **kwargs)

        # If cond_scale is 1, return the regular model output
        if cond_scale == 1:
            return logits

        # Otherwise, call the forward method again, but with the conditional dropout probability set to 1 (i.e. no conditioning)
        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        # Finally, the output is a linear interpolation between the regular model output and the null_logits, controlled by cond_scale
        # and the output will be a weighted combination of the two
        return null_logits + (logits - null_logits) * cond_scale


    def forward(self, x, time, *, lowres_cond_img = None, lowres_noise_times = None, text_embeds = None, text_mask = None, cond_images = None, self_cond = None, cond_drop_prob = 0.):
        '''
        Forward pass of the U-Net. The forward pass is split into two parts: the first part is the conditioning part, where the
        conditioning is processed and the conditioning tokens are derived. The second part is the actual U-Net, where the conditioning
        tokens are used to condition the synthesis.

        Parameters:
        (The parameters with a * are keyword-only arguments, and must be specified by name, not by position. All default to None or 0.)
        - x (tensor): The input tensor
        - time (tensor): The time tensor
        - lowres_cond_img (tensor): The low resolution conditioning image
        - lowres_noise_times (tensor): The low resolution noise times
        - text_embeds (tensor): The text embeddings
        - text_mask (tensor): The text mask
        - cond_images (tensor): The conditioning images
        - self_cond (tensor): The self conditioning
        - cond_drop_prob (float): The probability of dropping the conditioning

        Returns:
        - x (tensor): The output tensor
        '''
        # Gets the batch size and device (GPU/CPU) from the input tensor
        batch_size, device = x.shape[0], x.device

        # If the image is conditioned on itself, concatenate the self-conditioning (another version of the image) to the input
        if self.self_cond:
            # If the self-conditioning is not specified, set it to a tensor of zeros with the same shape as the input
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim = 1)

        # Check if low resolution conditioning is used, if it is, make sure the low resolution conditioning image and noise times are specified
        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)), 'low resolution conditioning noise time must be present'

        # If it exists, concatenate the low resolution conditioning image to the input, along the channel dimension
        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # condition on input image
        # Check if the model is conditioned on an input image, and if it is, make sure the conditioning image is specified
        assert not (self.has_cond_image ^ exists(cond_images)), 'you either requested to condition on an image on the unet, but the conditioning image is not supplied, or vice versa'

        # If the model is conditioned on an image 
        if exists(cond_images):
            # Check if the number of channels in the conditioning image matches the number of channels specified in the model
            assert cond_images.shape[1] == self.cond_images_channels, 'the number of channels on the conditioning image you are passing in does not match what you specified on initialiation of the unet'
            # Resize (interpolate) the conditioning image to the same size as the input image
            cond_images = resize_image_to(cond_images, x.shape[-1], mode = self.resize_mode)
            # Concatenate the conditioning image to the input
            x = torch.cat((cond_images, x), dim = 1)

        # Initial convolution with either cross embedding or normal convolution
        x = self.init_conv(x)

        # If the model is using residual connection from the initial convolution to the final convolution, save the initial convolution output
        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # Derive hidden representations of the time embedding
        time_hiddens = self.to_time_hiddens(time)

        # Project the time conditioning to multiple time tokens
        time_tokens = self.to_time_tokens(time_hiddens)
        # Derive the condensed time conditioning
        t = self.to_time_cond(time_hiddens)

        # add lowres time conditioning to time hiddens
        # and add lowres time tokens along sequence dimension for attention
        # If low resolution conditioning is used
        if self.lowres_cond:
            # Derive hidden representations of the low resolution time embedding
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            # Project the low resolution time conditioning to multiple time tokens
            lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
            # Derive the condensed low resolution time conditioning
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)

            # Add the low resolution time conditioning to the time conditioning
            t = t + lowres_t
            # Concatenate the low resolution time tokens to the time tokens
            time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim = -2)


        # TEXT CONDITIONING #

        # Variable to store the text tokens
        text_tokens = None

        # If the model is conditioned on text and text embeddings are provided
        if exists(text_embeds) and self.cond_on_text:

            # With a probability of cond_drop_prob, drop the conditioning, using a mask
            text_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device = device)

            # expand text keep mask to match text embeds
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            # expand text keep mask to match text hiddens
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')

            # Create text to tokens by projecting the text embeddings to the conditioning dimensionality
            text_tokens = self.text_to_cond(text_embeds)

            # If the maximum text length is specified, truncate the text tokens to the maximum text length
            text_tokens = text_tokens[:, :self.max_text_len]

            # If a text mask is provided, truncate the text mask to the maximum text length
            if exists(text_mask):
                text_mask = text_mask[:, :self.max_text_len]

            # Get the length of the text tokens
            text_tokens_len = text_tokens.shape[1]
            # Calculate the remainder of the maximum text length and the text tokens length
            remainder = self.max_text_len - text_tokens_len

            # If the remainder is greater than 0, pad the text tokens with zeros
            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            # If a text mask is provided and the remainder is greater than 0, pad the text mask with zeros
            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value = False)

                # Reshape the text mask to match the text tokens
                text_mask = rearrange(text_mask, 'b n -> b n 1')
                # Update the text keep mask to include the text mask, i.e. if the text mask is false, the text keep mask is false
                text_keep_mask_embed = text_mask & text_keep_mask_embed

            # Convert the null text embedding to the same data type as the text tokens
            null_text_embed = self.null_text_embed.to(text_tokens.dtype) # for some reason pytorch AMP not working

            # Replace the text tokens with the null text embedding where the text keep mask is false
            text_tokens = torch.where(
                text_keep_mask_embed,
                text_tokens,
                null_text_embed
            )

            # If the model is using attention pooling, pool over the text tokens
            if exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)

            # extra non-attention conditioning by projecting and then summing text embeddings to time
            # termed as text hiddens
            # Compute mean pooled text tokens as mean of text tokens 
            mean_pooled_text_tokens = text_tokens.mean(dim = -2)

            # Project the mean pooled text tokens to hiddens with non-attention
            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)

            # Convert the null text hidden to the same data type as the time tokens
            null_text_hidden = self.null_text_hidden.to(t.dtype)

            # Replace the text hiddens with the null text hidden where the text keep mask is false
            text_hiddens = torch.where(
                text_keep_mask_hidden,
                text_hiddens,
                null_text_hidden
            )

            # Add the text hiddens to the time conditioning
            t = t + text_hiddens

        # main conditioning tokens (c) (all conditioning is added to the time conditioning)
        # If the text tokens are specified, concatenate the text tokens to the time tokens
        c = time_tokens if not exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim = -2)

        # normalize conditioning tokens
        c = self.norm_cond(c)

        # initial resnet block (for memory efficient unet)
        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)

        ###################
        # UNET WALKTROUGH #
        ###################

        # go through the layers of the unet, down and up

        # List to store the hidden representations of the feature maps
        hiddens = []


        # DOWNSAMPLING #

        # Loop through the downsampling layers
        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            # If pre-downsampling is used, downsample the input
            if exists(pre_downsample):
                x = pre_downsample(x)

            # Pass the input through the initial block
            x = init_block(x, t, c)

            # Loop through the resnet blocks
            for resnet_block in resnet_blocks:
                # Pass the input through the resnet block
                x = resnet_block(x, t)
                # Append the hidden representation of the feature map to the list of hidden representations
                hiddens.append(x)

            # Pass the input through the attention block
            x = attn_block(x, c)
            # Append the hidden representation of the feature map to the list of hidden representations
            hiddens.append(x)

            # If post-downsampling is used, downsample the output
            if exists(post_downsample):
                x = post_downsample(x)


        # MIDDLE LAYERS #

        # Pass the input through the first middle resnet block
        x = self.mid_block1(x, t, c)

        # Pass the input through the attention block if it exists
        if exists(self.mid_attn):
            x = self.mid_attn(x)

        # Pass the input through the second middle resnet block
        x = self.mid_block2(x, t, c)

        # Define Add skip-connection function: 
        # Concatenate the last f-map from the hiddens (scaled by skip_connect_scale) to the current feature map
        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)
        

        # UPSAMPLING #

        # Initialize an empty list to store the hidden representations of the upsampled feature maps
        up_hiddens = []

        # Loop through the upsampling layers (reverse of downsampling layers) 
        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            # Add skip connection 
            x = add_skip_connection(x)
            # Pass the input through the initial block
            x = init_block(x, t, c)

            # Loop through the resnet blocks
            for resnet_block in resnet_blocks:
                # Add skip connection 
                x = add_skip_connection(x)
                # Pass the input through the resnet block
                x = resnet_block(x, t)

            # Pass the input through the attention block
            x = attn_block(x, c)
            # Append the current feature map
            up_hiddens.append(x.contiguous())
            x = upsample(x)

        # If the feature maps from the upsampling blocks are to be combined
        x = self.upsample_combiner(x, up_hiddens)

        # If set to add residual connection from initial conv output, do so here
        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim = 1)

        # If the final resnet block is used, pass the input through it
        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        # If low resolution conditioning is used, concatenate the low resolution conditioning image to the output
        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # Pass the output through the final convolution
        return self.final_conv(x)
    

class NullUnet(nn.Module):
    '''
    Placeholder/dummy class for the null U-Net. This class is used to allow for the same code to be used for both the regular U-Net 
    and the null U-Net.
    Used for testing purposes.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor method for the NullUnet class.

        Parameters:
        - args: Variable length argument list
        - kwargs: Arbitrary keyword arguments
        '''
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        # Set the low resolution conditioning flag to False
        self.lowres_cond = False
        # Define a single scalar value dummy parameter (required of an nn.Module) 
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def cast_model_parameters(self, *args, **kwargs):
        '''
        Method to mimic method of actual U-Net performing some kind of operation on parameters.
        Does nothing, just returns self.
        '''
        return self

    def forward(self, x, *args, **kwargs):
        '''
        Forward pass of the NullUnet. Does nothing, just returns the input.
        '''
        return x


# PREDEFINED UNETS, CONFIGURATIONS MATCHING APPENDIX OF PAPER #

class BaseUnet64(Unet):
    '''
    Class defining a base U-Net configuration.
    First unet in cascaded DDPM.
    Generates 64x64 images.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Constructor method for the BaseUnet64 class. 
        Calls the constructor of the Unet class with defined keyword arguments.
        '''
        default_kwargs = dict(
            dim = 512, # Base dimensionality of the model (i.e. the dimensionality of the first layer)
            dim_mults = (1, 2, 3, 4), # Dimensionality multipliers for each layer
            num_resnet_blocks = 3, # Number of resnet blocks
            layer_attns = (False, True, True, True), # Whether to use self-attention blocks at each layer
            layer_cross_attns = (False, True, True, True), # Whether to use cross-attention blocks at each layer
            attn_heads = 8, # Number of attention heads
            ff_mult = 2., # Multiplier for the feed-forward layer dimensions in the attention mechanisms
            memory_efficient = False # Whether to use checkpointing to reduce memory usage
        )

        # Call the constructor of the Unet class with the default keyword arguments
        super().__init__(*args, **{**default_kwargs, **kwargs})


class SRUnet256(Unet):
    '''
    Class defining a U-Net configuration for super-resolution from 64x64 to 256x256.
    Second unet in cascaded DDPM.
    Generates 256x256 images.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor method for the SRUnet256 class.
        Calls the constructor of the Unet class with defined keyword arguments.
        '''
        default_kwargs = dict(
            dim = 128, # Base dimensionality of the model (i.e. the dimensionality of the first layer)
            dim_mults = (1, 2, 4, 8), # Dimensionality multipliers for each layer
            num_resnet_blocks = (2, 4, 8, 8), # Number of resnet blocks
            layer_attns = (False, False, False, True), # Whether to use self-attention blocks at each layer
            layer_cross_attns = (False, False, False, True), # Whether to use cross-attention blocks at each layer
            attn_heads = 8, # Number of attention heads
            ff_mult = 2., # Multiplier for the feed-forward layer dimensions in the attention mechanisms
            memory_efficient = True # Whether to use checkpointing to reduce memory usage
        )

        # Call the constructor of the Unet class with the default keyword arguments
        super().__init__(*args, **{**default_kwargs, **kwargs})


class SRUnet1024(Unet):
    '''
    Class defining a U-Net configuration for super-resolution from 256x256 to 1024x1024.
    Third unet in cascaded DDPM.
    Generates 1024x1024 images.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Constructor method for the SRUnet1024 class.
        Calls the constructor of the Unet class with defined keyword arguments.
        '''
        default_kwargs = dict(
            dim = 128, # Base dimensionality of the model (i.e. the dimensionality of the first layer)
            dim_mults = (1, 2, 4, 8),   # Dimensionality multipliers for each layer
            num_resnet_blocks = (2, 4, 8, 8), # Number of resnet blocks
            layer_attns = False, # Whether to use self-attention blocks at each layer
            layer_cross_attns = (False, False, False, True), # Whether to use cross-attention blocks at each layer
            attn_heads = 8, # Number of attention heads
            ff_mult = 2., # Multiplier for the feed-forward layer dimensions in the attention mechanisms
            memory_efficient = True # Whether to use checkpointing to reduce memory usage
        )

        # Call the constructor of the Unet class with the default keyword arguments
        super().__init__(*args, **{**default_kwargs, **kwargs})


# MAIN IMAGEN DDPM CLASS: CASCADING DDPM FROM HO ET AL. #

class Imagen(nn.Module):
    '''
    Main class defining the Imagen cascading DDPM model.

    Methods:
    - __init__: Constructor method for the Imagen class
    - force_unconditional:
    - device:
    - get_unet: 
    - reset_unets_all_one_device:
    - one_unet_in_gpu:
    - state_dict:
    - load_state_dict:
    - p_mean_variance:
    - p_sample:
    - p_sample_loop:
    - sample:
    - p_losses:
    - forward: 
    '''
    def __init__(
        self,
        unets,                                      # Tuple of unets, each unet is a tuple of (klass, kwargs) - klass is the class of the unet, and kwargs are the keyword arguments to pass to the unet
        *,                                          # * All arguments after this are keyword-only arguments *
        image_sizes,                                # fFr cascading ddpm, image size at each stage
        text_encoder_name = DEFAULT_T5_NAME,        # What text encoder to use
        text_embed_dim = None,                      # Dimensionality of the text embeddings
        channels = 3,                               # Number of channels in the input image
        timesteps = 1000,                           # Number of timesteps in the diffusion process
        cond_drop_prob = 0.1,                       # Probability of dropping the conditioning
        loss_type = 'l2',                           # Type of loss to use, default is l2 (MSE)
        noise_schedules = 'cosine',                 # Noise schedule for each unet
        pred_objectives = 'noise',                  # Predicting noise or image
        random_crop_sizes = None,                   # Random crop sizes for each unet 
        lowres_noise_schedule = 'linear',           # Noise schedule for low resolution conditioning
        lowres_sample_noise_level = 0.2,            # (Augmenting conditioning) In the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        per_sample_random_aug_noise_level = False,  # Unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        condition_on_text = True,                   # Whether to condition on text
        auto_normalize_img = True,                  # Whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        dynamic_thresholding = True,                # Whether to use dynamic thresholding (helps controlling the range of the pixel values of the reconstructed image)
        dynamic_thresholding_percentile = 0.95,     # Percantile of the absolute values of reconstructed samples per batch element. Unsure what this was based on perusal of paper 
        only_train_unet_number = None,              # If you only want to train a specific unet, specify the index of the unet here
        temporal_downsample_factor = 1,             # Temporal downsample factor for each unet (used for video)
        resize_cond_video_frames = True,            # Whether to resize the conditioning video frames to the same size as the input video frames
        resize_mode = 'nearest',                    # Resize mode for resizing operations 
        min_snr_loss_weight = True,                 # https://arxiv.org/abs/2303.09556 - whether to use the min_snr_loss_weight
        min_snr_gamma = 5                           # https://arxiv.org/abs/2303.09556 - gamma for the min_snr_loss_weight
    ):
        # Call the constructor (the __init__ function) of the base class (nn.Module).
        super().__init__()

        # Set loss type
        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            # If the loss type is not one of the above, raise an error
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn


        # Conditioning hyperparameters
        self.condition_on_text = condition_on_text
        self.unconditional = not condition_on_text


        # Number of channels
        self.channels = channels


        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        # Cast to tuple if not already
        unets = cast_tuple(unets)
        # Get number of unets
        num_unets = len(unets)


        # Determine noise schedules per unet
        timesteps = cast_tuple(timesteps, num_unets)

        # Make sure noise schedule defaults to 'cosine', 'cosine', and then 'linear' for rest of super-resoluting unets
        noise_schedules = cast_tuple(noise_schedules)
        noise_schedules = pad_tuple_to_length(noise_schedules, 2, 'cosine')
        noise_schedules = pad_tuple_to_length(noise_schedules, num_unets, 'linear')
        
        # Define noise scheduler class
        noise_scheduler_klass = GaussianDiffusionContinuousTimes
        self.noise_schedulers = nn.ModuleList([])

        # Construct noise schedulers for each unet
        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = noise_scheduler_klass(noise_schedule = noise_schedule, timesteps = timestep)
            self.noise_schedulers.append(noise_scheduler)


        # Randomly cropping for upsampler training. Raise an error if the base unet is also being randomly cropped
        self.random_crop_sizes = cast_tuple(random_crop_sizes, num_unets)
        assert not exists(first(self.random_crop_sizes)), 'you should not need to randomly crop image during training for base unet, only for upsamplers - so pass in `random_crop_sizes = (None, 128, 256)` as example'

        # Lowres augmentation noise schedule - used for augmenting the low resolution conditioning image
        self.lowres_noise_schedule = GaussianDiffusionContinuousTimes(noise_schedule = lowres_noise_schedule)


        # DDPM objectives - predicting noise by default
        self.pred_objectives = cast_tuple(pred_objectives, num_unets)


        # Get text encoder and text embedding dimensionality 
        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))

        # Define function to encode text, partial function of t5_encode_text
        self.encode_text = partial(t5_encode_text, name = text_encoder_name)


        # Construct unets
        self.unets = nn.ModuleList([])

        # Keep track of which unet is being trained
        self.unet_being_trained_index = -1 
        # Keep track of whether to only train a specific unet
        self.only_train_unet_number = only_train_unet_number

        # Loop through the unets
        for ind, one_unet in enumerate(unets):
            # Make sure the unets are Unet instances
            assert isinstance(one_unet, (Unet, Unet3D, NullUnet))
            
            is_first = ind == 0

            # Make an instance of the unet eith the specified keyword arguments              
            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                cond_on_text = self.condition_on_text,
                text_embed_dim = self.text_embed_dim if self.condition_on_text else None,
                channels = self.channels,
                channels_out = self.channels
            )

            # Append the unet to the list of unets
            self.unets.append(one_unet)


        # Unet image sizes
        image_sizes = cast_tuple(image_sizes)
        self.image_sizes = image_sizes

        # Make sure the number of unets matches the number of image sizes
        assert num_unets == len(image_sizes), f'you did not supply the correct number of u-nets ({len(unets)}) for resolutions {image_sizes}'

        # Define the number of channels for samples as the number of channels in the input image of each unet
        self.sample_channels = cast_tuple(self.channels, num_unets)

        # Determine whether we are training on images or video
        is_video = any([isinstance(unet, Unet3D) for unet in self.unets])
        self.is_video = is_video


        # Define function to pad dimensions to datatype
        self.right_pad_dims_to_datatype = partial(rearrange, pattern = ('b -> b 1 1 1' if not is_video else 'b -> b 1 1 1 1'))


        # Define function to resize image to specified size (interpolation mode is nearest by default)
        self.resize_to = resize_video_to if is_video else resize_image_to
        self.resize_to = partial(self.resize_to, mode = resize_mode)


        # Temporal interpolation for video
        temporal_downsample_factor = cast_tuple(temporal_downsample_factor, num_unets)
        self.temporal_downsample_factor = temporal_downsample_factor

        self.resize_cond_video_frames = resize_cond_video_frames
        self.temporal_downsample_divisor = temporal_downsample_factor[0]

        assert temporal_downsample_factor[-1] == 1, 'downsample factor of last stage must be 1'
        assert tuple(sorted(temporal_downsample_factor, reverse = True)) == temporal_downsample_factor, 'temporal downsample factor must be in order of descending'


        # Cascading ddpm related stuff
        # Get the low resolutiom conditioning flags and make sure the first unet is unconditioned
        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'


        # Augmenting conditioning noise level
        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level


        # Classifier free guidance (conditional dropout and flag for whether to use it)
        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.


        # Normalize and unnormalize image functions
        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)


        # Dynamic thresholding
        self.dynamic_thresholding = cast_tuple(dynamic_thresholding, num_unets)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile


        # Min snr loss weight and gamma
        min_snr_loss_weight = cast_tuple(min_snr_loss_weight, num_unets)
        min_snr_gamma = cast_tuple(min_snr_gamma, num_unets)

        # Make sure the length of the min snr loss weight and gamma are the same as the number of unets
        assert len(min_snr_loss_weight) == len(min_snr_gamma) == num_unets
        # Save gammas in tuple if min snr loss weight is True, else save as None
        self.min_snr_gamma = tuple((gamma if use_min_snr else None) for use_min_snr, gamma in zip(min_snr_loss_weight, min_snr_gamma))


        # One temp parameter for keeping track of device 
        self.register_buffer('_temp', torch.tensor([0.]), persistent = False)


        # Default to device of unets passed in
        self.to(next(self.unets.parameters()).device)


    def force_unconditional_(self):
        '''
        Method to force the model to be unconditional (on text, the Unets =! base-unet are still conditioned on previous output).
        '''
        # Set the conditional on text flag to False and the unconditional flag to True
        self.condition_on_text = False
        self.unconditional = True

        # Loop through the unets and set the conditional on text flag to False
        for unet in self.unets:
            unet.cond_on_text = False

    @property
    def device(self):
        '''
        Getter method for the device property. Makes it possible to get the device of the model by calling `model.device` instead of `model._temp.device`.
        '''
        return self._temp.device


    def get_unet(self, unet_number):
        '''
        Method to get the unet at the specified index.

        Parameters:
        - unet_number (int): The index of the unet to get

        Returns:
        - unet (Unet): The unet at the specified index
        '''
        # Make sure the unet number is valid and get the index
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        # Check if unets is a ModuleList
        if isinstance(self.unets, nn.ModuleList):
            # If it is, make it to a list and delete the attribute
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            # Set the unets to the list
            self.unets = unets_list

        # If the unet of interest is not the unet being trained
        if index != self.unet_being_trained_index:
            # Loop through the unets
            for unet_index, unet in enumerate(self.unets):
                # set the unet to the device if it is the correct unet 
                unet.to(self.device if unet_index == index else 'cpu')

        # Set the unet being trained index to the index
        self.unet_being_trained_index = index

        # Return the unet at the index
        return self.unets[index]


    def reset_unets_all_one_device(self, device = None):
        '''
        Update where U-Nets are stored, and reset the U-Nets to be on the same device.
        '''
        # If the device is not specified, set it to the device of the model
        device = default(device, self.device)
        # Store the unets in a ModuleList. * required to unpack the list
        self.unets = nn.ModuleList([*self.unets])
        # Set the unets to the device
        self.unets.to(device)

        # Set the unet being trained index to -1
        self.unet_being_trained_index = -1

    # Context manager is a construct allowing you to set up a context, run some code, and then tear down the context, regardless of whether the code succeeded or not. 
    # Often used for setting up and tearing down a GPU context, or a database connection, or a file handle, etc.
    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        '''
        Context manager, that temporarily moves specified UNet to GPU for duration of context, while moving all other UNets to CPU.
        Useful for savin GPU memory when training on multiple UNets.
        Runs code within a 'with' statement.

        USAGE:
        with some_instance.one_unet_in_gpu(unet_number=3):
            # Code here can use the third U-Net on the GPU
            # All other U-Nets are on the CPU
        # Outside the block, all U-Nets are back to their original devices

        Parameters:
        - unet_number (int): The index of the unet to move to the GPU
        - unet (Unet): The unet to move to the GPU
        '''
        # Make sure either the unet number or the unet is specified
        assert exists(unet_number) ^ exists(unet)

        # If the unet number is specified, get the unet at the index
        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        # Get the cpu device object
        cpu = torch.device('cpu')

        # List of all the devices U-Nets are stored on currently
        devices = [module_device(unet) for unet in self.unets]

        # Move all U-Nets to CPU
        self.unets.to(cpu)
        # Move the specified U-Net to GPU
        unet.to(self.device)

        # Yield: the point at which the code in the 'with' statement is executed
        yield

        # Move all U-Nets back to their original devices
        for unet, device in zip(self.unets, devices):
            unet.to(device)


    def state_dict(self, *args, **kwargs):
        '''
        Overwriting the state_dict function to reset the U-Nets to be on the same device.
        Ensures that all parameters in the model are refferenced correctly when saved.

        Parameters:
        - args: Variable length argument list
        - kwargs: Arbitrary keyword arguments
        '''
        # Reset the U-Nets to be on the same device
        self.reset_unets_all_one_device()
        # Return the state dictionary
        return super().state_dict(*args, **kwargs)


    def load_state_dict(self, *args, **kwargs):
        '''
        Overwriting the load_state_dict function to reset the U-Nets to be on the same device.
        Ensures that all parameters in the model are refferenced correctly when loaded.

        Parameters:
        - args: Variable length argument list
        - kwargs: Arbitrary keyword arguments
        '''
        # Reset the U-Nets to be on the same device
        self.reset_unets_all_one_device()
        # Return the loaded state dictionary
        return super().load_state_dict(*args, **kwargs)

    # gaussian diffusion methods

    def p_mean_variance(
        self,                                       #  
        unet,                                       # Unet to use to estimate noise or signal in image at timestep t
        x,                                          # Current noisy image
        t,                                          # Current timestep in denoising process
        *,                                          # * All arguments after this are keyword-only arguments *
        noise_scheduler,                            # Noise scheduler 
        text_embeds = None,                         # Text embeddings
        text_mask = None,                           # Text mask
        cond_images = None,                         # Conditioning images
        cond_video_frames = None,                   # Conditioning video frames
        post_cond_video_frames = None,              # Post conditioning video frames
        lowres_cond_img = None,                     # Low resolution conditioning image
        self_cond = None,                           # Self conditioning
        lowres_noise_times = None,                  # Low resolution noise times
        cond_scale = 1.,                            # Conditioning scale
        model_output = None,                        # Model output 
        t_next = None,                              # Next timestep in denoising process
        pred_objective = 'noise',                   # Predicting noise or image
        dynamic_threshold = True                    # Whether to use dynamic thresholding
    ):
        '''
        Method to estimate the mean and variance for the distribution that the model believes the 'denoised' image (at time 't') to be drawn from.

        Returns:
        - mean_and_variance (tuple): Tuple containing the mean and variance of the distribution that the model believes the 'denoised' image to be drawn from
        - x_start (tensor): Estimate of the original image
        '''
        # Check if classifier free guidance can be used
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        # If the model is working with videos, it sets the keyword arguments for video accordingly
        video_kwargs = dict()
        if self.is_video:
            video_kwargs = dict(
                cond_video_frames = cond_video_frames,
                post_cond_video_frames = post_cond_video_frames,
            )

        # The prediction is set to the output, unless not available. Then using the unet method to generate prediction
        pred = default(model_output, lambda: unet.forward_with_cond_scale(
            x,                                      # Current noisy image
            noise_scheduler.get_condition(t),       # Current noise level at timestep t
            text_embeds = text_embeds,              # Text embeddings
            text_mask = text_mask,                  # Text mask
            cond_images = cond_images,              # Conditioning images
            cond_scale = cond_scale,                # Conditioning scale
            lowres_cond_img = lowres_cond_img,      # Low resolution conditioning image
            self_cond = self_cond,                  # Self conditioning
            lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_noise_times), # Low resolution noise times
            **video_kwargs                          # Video keyword arguments, if applicable
        ))

        # Estimate x_start (estimate of original image), either from (x: current noisy image, t: current timestep, noise: current noise level) or from 'v'
        if pred_objective == 'noise':
            x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)
        elif pred_objective == 'x_start':
            x_start = pred
        elif pred_objective == 'v':
            x_start = noise_scheduler.predict_start_from_v(x, t = t, v = pred)
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        # If dynamic thresholding is used, calculate the dynamic threshold
        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            s = torch.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.dynamic_thresholding_percentile,
                dim = -1
            )

            # clamp s to be at least 1
            s.clamp_(min = 1.)
            # Pad s to be the same shape as x_start
            s = right_pad_dims_to(x_start, s)
            # Divide x_start by s to get the dynamic threshold 
            x_start = x_start.clamp(-s, s) / s
        else:
            # If dynamic thresholding is not used, clamp x_start to be between -1 and 1
            x_start.clamp_(-1., 1.)

        # Return the mean and variance (q posterior) of the noise level at timestep t, and the estimate of the original image x_start
        mean_and_variance = noise_scheduler.q_posterior(x_start = x_start, x_t = x, t = t, t_next = t_next)
        return mean_and_variance, x_start

    # Set the model to evaluation mode
    @torch.no_grad()
    def p_sample(
        self,                                       #   
        unet,                                       # UNet to estimate parameters of posterior distribution of the clean image given current noisy image
        x,                                          # Current noisy image
        t,                                          # Current timestep in reverse diffusion process
        *,
        noise_scheduler,                            # Noise scheduler
        t_next = None,                              # Next timestep in reverse diffusion process
        text_embeds = None,                         # Text embeddings
        text_mask = None,                           # Text mask
        cond_images = None,                         # Conditioning images
        cond_video_frames = None,                   # Conditioning video frames
        post_cond_video_frames = None,              # Post conditioning video frames
        cond_scale = 1.,                            # Conditioning scale
        self_cond = None,                           # Self conditioning
        lowres_cond_img = None,                     # Low resolution conditioning image
        lowres_noise_times = None,                  # Low resolution noise times
        pred_objective = 'noise',                   # Predicting noise or image
        dynamic_threshold = True                    # Whether to use dynamic thresholding
    ):
        '''
        Method to sample from the estimated posterior distribution of the clean image given current noisy image. 
        Performs single denoising step.
        Generates a single sample ('pred') at a particular time step 't' based on current state of the noisy image 'x'.
        Also returns the estimate of the original image 'x_start'.

        '''
        # Get the shape and device of the current noisy image
        b, *_, device = *x.shape, x.device

        # If the model is working with videos, it sets the keyword arguments for video accordingly
        video_kwargs = dict()
        if self.is_video:
            video_kwargs = dict(
                cond_video_frames = cond_video_frames,
                post_cond_video_frames = post_cond_video_frames,
            )

        # Get the mean and variance of the noise level at timestep t, and the estimate of the original image x_start
        (model_mean, _, model_log_variance), x_start = self.p_mean_variance(
            unet,
            x = x,
            t = t,
            t_next = t_next,
            noise_scheduler = noise_scheduler,
            text_embeds = text_embeds,
            text_mask = text_mask,
            cond_images = cond_images,
            cond_scale = cond_scale,
            lowres_cond_img = lowres_cond_img,
            self_cond = self_cond,
            lowres_noise_times = lowres_noise_times,
            pred_objective = pred_objective,
            dynamic_threshold = dynamic_threshold,
            **video_kwargs
        )

        # Sample random noise
        noise = torch.randn_like(x)

        # No noise when t == 0 (check if t_next == 0 if using continuous noise scheduler else check if t == 0) 
        is_last_sampling_timestep = (t_next == 0) if isinstance(noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        # If it is the last sampling timestep, set the noise to 0
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # Make prediction as the mean + the scaled noise (scaled by the variance)
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Return the prediction and the estimate of the original image x_start
        # Pred is the prediction at timestep t, x_start is the estimate of the original image
        return pred, x_start

    # Set the model to evaluation mode
    @torch.no_grad()
    def p_sample_loop(
        self,
        unet,                                       # UNet to estimate from
        shape,                                      # Shape of the image to generate
        *,
        noise_scheduler,                            # Noise scheduler
        lowres_cond_img = None,                     # Low resolution conditioning image
        lowres_noise_times = None,                  # Low resolution noise times
        text_embeds = None,                         # Text embeddings
        text_mask = None,                           # Text mask
        cond_images = None,                         # Conditioning images
        cond_video_frames = None,                   # Conditioning video frames
        post_cond_video_frames = None,              # Post conditioning video frames
        inpaint_images = None,                      # Inpainting images (inpainting is filling a part of an image with new content generated by model - used for super-resolution)
        inpaint_videos = None,                      # Inpainting videos
        inpaint_masks = None,                       # Inpainting masks
        inpaint_resample_times = 5,                 # Inpainting resample times
        init_images = None,                         # Initial images
        skip_steps = None,                          # Skip steps 
        cond_scale = 1,                             # Conditioning scale
        pred_objective = 'noise',                   # Predicting noise or image
        dynamic_threshold = True,                   # Whether to use dynamic thresholding
        use_tqdm = True                             # Whether to use tqdm
    ):
        '''
        Method to sample from the estimated posterior distribution of the clean image given current noisy image. 
        Performs full denoising process, from timestep T to timestep 0.

        Method to generate images from noise by reversing diffusion process using trained UNet(s). 
        Can be conditioned on text, images, or videos.
        Can also be used for super-resolution (inpainting) by passing in inpainting images and masks.

        Returns:
        - unnormalize_img (tensor): The generated image(s) - one image per batch element
        '''
        # Get devixe
        device = self.device

        # Get batch size
        batch = shape[0]
        # Create random noise image of desired shape. Set to device
        img = torch.randn(shape, device = device)

        # Check if is video
        is_video = len(shape) == 5
        # Get number of frames
        frames = shape[-3] if is_video else None
        # Resize keyword arguments
        resize_kwargs = dict(target_frames = frames) if exists(frames) else dict()

        # For initialization with an image or video
        if exists(init_images):
            # Add the initial image to the random noise image
            img += init_images

        # Keep track of x0, for self conditioning
        x_start = None

        # Prepare inpainting
        inpaint_images = default(inpaint_videos, inpaint_images)

        # Check if inpainting is being used
        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        # Set inpainting resample times
        resample_times = inpaint_resample_times if has_inpainting else 1

        if has_inpainting:
            # Normalize and resize inpainting images
            inpaint_images = self.normalize_img(inpaint_images)
            inpaint_images = self.resize_to(inpaint_images, shape[-1], **resize_kwargs)
            inpaint_masks = self.resize_to(rearrange(inpaint_masks, 'b ... -> b 1 ...').float(), shape[-1], **resize_kwargs).bool()

        # Get the noise sampling timesteps 
        timesteps = noise_scheduler.get_sampling_timesteps(batch, device = device)

        # Whether to skip any steps (for example, if you want to skip the first 1000 steps) and remove them from the timesteps
        skip_steps = default(skip_steps, 0)
        timesteps = timesteps[skip_steps:]

        # Video conditioning kwargs
        video_kwargs = dict()
        if self.is_video:
            video_kwargs = dict(
                cond_video_frames = cond_video_frames,
                post_cond_video_frames = post_cond_video_frames,
            )

        # Loop through the timesteps and samples drawn from the noise_scheduler, using tqdm to show progress bar
        for times, times_next in tqdm(timesteps, desc = 'sampling loop time step', total = len(timesteps), disable = not use_tqdm):
            # Check if next timestep is last timestep
            is_last_timestep = times_next == 0

            # Go through resample times reversed (going actually through the reversed diffusion process)
            for r in reversed(range(resample_times)):
                # Check if last resample step
                is_last_resample_step = r == 0

                # If inpainting is used
                if has_inpainting:
                    # Sample noised inpaint images at desired timesteps
                    noised_inpaint_images, *_ = noise_scheduler.q_sample(inpaint_images, t = times)
                    # Return the images with the inpainting masks (inpainting masks are used to determine which pixels to inpaint)
                    img = img * ~inpaint_masks + noised_inpaint_images * inpaint_masks

                # If self conditioning is used, condition on the previous output (x_start)
                self_cond = x_start if unet.self_cond else None

                # Sample current timesteo image and estimate of original image from the model
                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    t_next = times_next,
                    text_embeds = text_embeds,
                    text_mask = text_mask,
                    cond_images = cond_images,
                    cond_scale = cond_scale,
                    self_cond = self_cond,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold,
                    **video_kwargs
                )

                # If inpainting is used and it is not the last resample step or the last timestep
                if has_inpainting and not (is_last_resample_step or torch.all(is_last_timestep)):
                    # Resample the noisy image multiple times to refine inpainted regions. 
                    renoised_img = noise_scheduler.q_sample_from_to(img, times_next, times)

                    # Replace the inpainted pixels with the renoised image
                    img = torch.where(
                        self.right_pad_dims_to_datatype(is_last_timestep),
                        img,
                        renoised_img
                    )

        # Clamp the image to be between -1 and 1
        img.clamp_(-1., 1.)

        # Final inpainting
        if has_inpainting:
            # Final image is combination of the inpainted regiond and the newly generated content.
            img = img * ~inpaint_masks + inpaint_images * inpaint_masks

        # Unnormalize the image
        unnormalize_img = self.unnormalize_img(img)
        # Return the sample
        return unnormalize_img

    # Set the model to evaluation mode
    @torch.no_grad()
    # Decorator to evaluate the function
    @eval_decorator
    # Bear type to check the types of the arguments
    @beartype
    def sample(
        self,
        texts: List[str] = None,                    # Text to condition on
        text_masks = None,                          # Text masks
        text_embeds = None,                         # Text embeddings
        video_frames = None,                        # Video frames 
        cond_images = None,                         # Conditioning images
        cond_video_frames = None,                   # Conditioning video frames
        post_cond_video_frames = None,              # Post conditioning video frames
        inpaint_videos = None,                      # Inpainting videos
        inpaint_images = None,                      # Inpainting images
        inpaint_masks = None,                       # Inpainting masks
        inpaint_resample_times = 5,                 # Inpainting resample times
        init_images = None,                         # Initial images
        skip_steps = None,                          # Time steps to skip
        batch_size = 1,                             # Batch size
        cond_scale = 1.,                            # Conditioning scale
        lowres_sample_noise_level = None,           # Low resolution sample noise level
        start_at_unet_number = 1,                   # What unet to start at
        start_image_or_video = None,                # Starting image or video
        stop_at_unet_number = None,                 # What unet to stop at
        return_all_unet_outputs = False,            # Whether to return all unet outputs
        return_pil_images = False,                  # Whether to return PIL images
        device = None,                              # Device to use
        use_tqdm = True,                            # Whether to use tqdm
        use_one_unet_in_gpu = True                  # Whether to use one unet at a time in the GPU
    ):
        '''
        Sample full outputs from the model. 
        '''
        # Set the device
        device = default(device, self.device)
        # Reset the unets to be on the same device
        self.reset_unets_all_one_device(device = device)

        # Cast image types to float if images exists
        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        # If conditioning text exists but not text embeddings, encode the text (as long as the model is not unconditional)
        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            # Make sure the text is not empty
            assert all([*map(len, texts)]), 'text cannot be empty'

            # Disabling autocast to prevent OOM errors (autocast chooses the optimal dtype for each operation to improve performance)
            with autocast(enabled = False):
                # Encode the text using the model's text encoder and return atention mask as well
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask = True)
            
            # Set the text embeddings to the device
            text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))

        # If the model is not unconditional
        if not self.unconditional:
            # Make sure the text embeddings are not empty
            assert exists(text_embeds), 'text must be passed in if the network was not trained without text `condition_on_text` must be set to `False` when training'

            # The text masks are set to the default text masks if not any
            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))
            # Get the batch size from the text embeddings
            batch_size = text_embeds.shape[0]

        # inpainting
        # Use inpaint videos if they exist, otherwise use inpaint images
        inpaint_images = default(inpaint_videos, inpaint_images)

        # If inpainting images exist 
        if exists(inpaint_images):
            # If model is unconditional
            if self.unconditional:
                # If batch size is 1, set the batch size to the number of inpainting images
                if batch_size == 1: # assume researcher wants to broadcast along inpainted images
                    batch_size = inpaint_images.shape[0]

            # If not unconditional, make sure that the batch size is equal to the number of inpainting images
            assert inpaint_images.shape[0] == batch_size, 'number of inpainting images must be equal to the specified batch size on sample `sample(batch_size=<int>)``'
            # Make sure that, if conditioned on text, the number of inpainting images is equal to the number of text embeddings
            assert not (self.condition_on_text and inpaint_images.shape[0] != text_embeds.shape[0]), 'number of inpainting images must be equal to the number of text to be conditioned on'

        # Make sure that, if conditioned on text, that the text embeddings are not empty
        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into imagen if specified'
        # Make sure, if not conditioned on text, that the text embeddings are empty$
        assert not (not self.condition_on_text and exists(text_embeds)), 'imagen specified not to be conditioned on text, yet it is presented'
        # Make sure, that if text embeddings exist, thet the embedding dimension is equal to the model's text embedding dimension
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        # Make sure that both inpaint images and masks are passed in
        assert not (exists(inpaint_images) ^ exists(inpaint_masks)),  'inpaint images and masks must be both passed in to do inpainting'

        # List to store outputs in
        outputs = []

        # Retrieve first (using 'next') parameter of model ('self.parameters()') and check if it is on the GPU
        is_cuda = next(self.parameters()).is_cuda
        # Get the device of the model
        device = next(self.parameters()).device

        # Get the noise level for the low resolution sample, set it to the default if not specified
        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        # Get the number of unets
        num_unets = len(self.unets)

        # Cast the conditional scaling and number of unets to a tuple
        cond_scale = cast_tuple(cond_scale, num_unets)

        # Add frame dimension for video
        # If model works with videos, and inpaint images exist
        if self.is_video and exists(inpaint_images):
            # Get the number of frames
            video_frames = inpaint_images.shape[2]

            # If the inpaint images are 3D, add a frame dimension
            if inpaint_masks.ndim == 3:
                inpaint_masks = repeat(inpaint_masks, 'b h w -> b f h w', f = video_frames)

            # Make sure that the inpaint masks first dimension is equal to the video frames 
            assert inpaint_masks.shape[1] == video_frames

        # If video does not exists raise an error. must be passed on sample time.
        assert not (self.is_video and not exists(video_frames)), 'video_frames must be passed in on sample time if training on video'

        # Compute all dimensions of frames
        all_frame_dims = calc_all_frame_dims(self.temporal_downsample_factor, video_frames)

        # Create keyword arguments for resizign operation where number of frames only defined if they exist. otherwise an empty dictionary is passed in
        frames_to_resize_kwargs = lambda frames: dict(target_frames = frames) if exists(frames) else dict()


        # Cast initial images and number of unets to a tuple
        init_images = cast_tuple(init_images, num_unets)
        # Normalize initial images if they exist
        init_images = [maybe(self.normalize_img)(init_image) for init_image in init_images]

        # Cast skip steps and number of unets to a tuple
        skip_steps = cast_tuple(skip_steps, num_unets)

        
        # Handle starting at a unet greater than 1, for training only-upscaler training
        # Check if start UNet is larger than one (no base UNet)
        if start_at_unet_number > 1:
            # Make sure that the start UNet is less than the number of unets
            assert start_at_unet_number <= num_unets, 'must start a unet that is less than the total number of unets'
            # Make sure that the UNet to start with is not equal to or less than the UNet to stop at
            assert not exists(stop_at_unet_number) or start_at_unet_number <= stop_at_unet_number
            # Make sure that the start image or video is passed in
            assert exists(start_image_or_video), 'starting image or video must be supplied if only doing upscaling'

            # Get the image size of the output of the previous UNet
            prev_image_size = self.image_sizes[start_at_unet_number - 2]
            # Get the frame size of the output of the previous UNet if working with videos
            prev_frame_size = all_frame_dims[start_at_unet_number - 2][0] if self.is_video else None
            # Resize the starting image or video to the image size of the output of the previous UNet
            img = self.resize_to(start_image_or_video, prev_image_size, **frames_to_resize_kwargs(prev_frame_size))


        # Go through each unet in cascade
        for unet_number, unet, channel, image_size, frame_dims, noise_scheduler, pred_objective, dynamic_threshold, unet_cond_scale, unet_init_images, unet_skip_steps in \
            tqdm(zip(range(1, num_unets + 1), self.unets, self.sample_channels, self.image_sizes, all_frame_dims, self.noise_schedulers, self.pred_objectives, self.dynamic_thresholding, cond_scale, init_images, skip_steps), disable = not use_tqdm):

            # If the current UNet is less than what we want to start at, skip it
            if unet_number < start_at_unet_number:
                continue
            
            # Make sure the current UNet is not Null
            assert not isinstance(unet, NullUnet), 'one cannot sample from null / placeholder unets'

            # Set the context to use one UNet in the GPU if specified and if the model is on the GPU
            context = self.one_unet_in_gpu(unet = unet) if is_cuda and use_one_unet_in_gpu else nullcontext()

            # With the given context of using only one UNet in the GPU
            with context:
                
                # If the model runs on video, set a dict with relevant kwargs
                video_kwargs = dict()
                if self.is_video:
                    video_kwargs = dict(
                        cond_video_frames = cond_video_frames,
                        post_cond_video_frames = post_cond_video_frames,
                    )

                    # Remove elements of dict that are None
                    video_kwargs = compact(video_kwargs)

                # If runs with video ande resize condition video frames is true
                if self.is_video and self.resize_cond_video_frames:
                    # Get the temporal downsample scale for the current UNet
                    downsample_scale = self.temporal_downsample_factor[unet_number - 1]

                    # Create a function to downsample the video frames by the downsample scale
                    temporal_downsample_fn = partial(scale_video_time, downsample_scale = downsample_scale)

                    # If the kwargs dict exists, transform the condition video frames and post condition video frames by the temporal downsample function
                    video_kwargs = maybe_transform_dict_key(video_kwargs, 'cond_video_frames', temporal_downsample_fn)
                    video_kwargs = maybe_transform_dict_key(video_kwargs, 'post_cond_video_frames', temporal_downsample_fn)

                # LOW RESOLUTION CONDITIONING #
                # Set low res cond image and low res noise times to none
                lowres_cond_img = lowres_noise_times = None
                # Get shape of the data. * is used to unpack the shape
                shape = (batch_size, channel, *frame_dims, image_size, image_size)

                # Make a dict for the kwargs containing the target frames if the model is working with videos 
                resize_kwargs = dict(target_frames = frame_dims[0]) if self.is_video else dict()

                # If UNet is conditioned on low resolution images (same image as img but with lower resolution)
                if unet.lowres_cond:
                    # Get low resolution noise times
                    lowres_noise_times = self.lowres_noise_schedule.get_times(batch_size, lowres_sample_noise_level, device = device)

                    # Resize image to low resolution
                    lowres_cond_img = self.resize_to(img, image_size, **resize_kwargs)

                    # Normalize low resolution image
                    lowres_cond_img = self.normalize_img(lowres_cond_img)
                    # Sample a noised image from the low resolution image at the low resolution noise times
                    lowres_cond_img, *_ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_noise_times, noise = torch.randn_like(lowres_cond_img))

                # If the UNet is conditioned on initial images or video
                if exists(unet_init_images):
                    # Resize the initial images or video to the image size of the current UNet
                    unet_init_images = self.resize_to(unet_init_images, image_size, **resize_kwargs)

                # Get shape of current stage
                shape = (batch_size, self.channels, *frame_dims, image_size, image_size)

                # Sample an image gone through the entire back-diffusion process with the relevant defined arguments
                img = self.p_sample_loop(
                    unet,
                    shape,
                    text_embeds = text_embeds,
                    text_mask = text_masks,
                    cond_images = cond_images,
                    inpaint_images = inpaint_images,
                    inpaint_masks = inpaint_masks,
                    inpaint_resample_times = inpaint_resample_times,
                    init_images = unet_init_images,
                    skip_steps = unet_skip_steps,
                    cond_scale = unet_cond_scale,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold,
                    use_tqdm = use_tqdm,
                    **video_kwargs
                )

                # Append image to outputs
                outputs.append(img)

            # If the current UNet is the UNet to stop at, break
            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        # Return all outputs if specified, otherwise return the last output
        output_index = -1 if not return_all_unet_outputs else slice(None)

        # Return the PIL images if specified, otherwise return the outputs
        if not return_pil_images:
            return outputs[output_index]

        # If not return all unet outputs, return only the last output
        if not return_all_unet_outputs:
            outputs = outputs[-1:]

        # Raise error that video conversion is not supported yet
        assert not self.is_video, 'converting sampled video tensor to video file is not supported yet'

        # Convert the outputs to PIL images
        pil_images = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim = 0))), outputs))

        # Return the PIL images for the outpux indeces relevant
        return pil_images[output_index] # now you have a bunch of pillow images you can just .save(/where/ever/you/want.png)


    # Bear type to check the types of the arguments
    @beartype
    def p_losses(
        self,                                       #
        unet: Union[Unet, Unet3D, NullUnet, DistributedDataParallel],   # Define the dtypes of UNet available
        x_start,                                    # Starting image, estimate of original image ?
        times,                                      # Timesteps in reverse diffusion process
        *,
        noise_scheduler,                            # Noise scheduler from diffusion utils
        lowres_cond_img = None,                     # Low resolution conditioning image
        lowres_aug_times = None,                    # Low resolution augmentation times
        text_embeds = None,                         # Text embeddings
        text_mask = None,                           # Text mask
        cond_images = None,                         # Conditioning images
        noise = None,                               # Noise
        times_next = None,                          # Next timestep in reverse diffusion process
        pred_objective = 'noise',                   # Predicting noise or image
        min_snr_gamma = None,                       # Minimum signal to noise ratio
        random_crop_size = None,                    # Random cropping size
        **kwargs
    ):
        '''
        Method to compute the training losses for a given UNet model during diffusion provess
        '''
        # If xstart has 5 dimensions, it is a video
        is_video = x_start.ndim == 5

        # Set noise to input noise or random 2D noise if not specified
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Normalize input image
        x_start = self.normalize_img(x_start)
        # Normalize low resolution conditioning image if it exists
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # random cropping during training
        # for upsamplers
        # If randpom crop size exists
        if exists(random_crop_size):
            # If the model is working with videos
            if is_video:
                # Get the number of frames
                frames = x_start.shape[2]
                # Rearrange the images to be in the format (batch, frames, channels, height, width)
                x_start, lowres_cond_img, noise = map(lambda t: rearrange(t, 'b c f h w -> (b f) c h w'), (x_start, lowres_cond_img, noise))

            # Create a random crop augmentation function with the random crop size
            aug = K.RandomCrop((random_crop_size, random_crop_size), p = 1.)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            
            # Augment the starting image, low resolution conditioning image, and noise
            x_start = aug(x_start)
            lowres_cond_img = aug(lowres_cond_img, params = aug._params)
            noise = aug(noise, params = aug._params)

            # If the model is working with videos, rearrange the images back to the original format
            if is_video:
                x_start, lowres_cond_img, noise = map(lambda t: rearrange(t, '(b f) c h w -> b c f h w', f = frames), (x_start, lowres_cond_img, noise))

        # Sample the noised image at time t, x_t
        x_noisy, log_snr, alpha, sigma = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)

        # also noise the lowres conditioning image
        # at sample time, they then fix the noise level of 0.1 - 0.3
        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy, *_ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_aug_times, noise = torch.randn_like(lowres_cond_img))

        # Get noise condition at times 
        noise_cond = noise_scheduler.get_condition(times)

        # UNet kwargs
        unet_kwargs = dict(
            text_embeds = text_embeds,
            text_mask = text_mask,
            cond_images = cond_images,
            lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_aug_times),
            lowres_cond_img = lowres_cond_img_noisy,
            cond_drop_prob = self.cond_drop_prob,
            **kwargs
        )

        # Set the self conditioning if it exists

        # Because 'unet' can be an instance of DistributedDataParallel coming from the
        # ImagenTrainer.unet_being_trained when invoking ImagenTrainer.forward(), we need to
        # access the member 'module' of the wrapped unet instance.
        self_cond = unet.module.self_cond if isinstance(unet, DistributedDataParallel) else unet.self_cond

        # If self conditioning exists and the random number is less than 0.5
        if self_cond and random() < 0.5:
            # Set the model to evaluation mode
            with torch.no_grad():
                # Get the prediction from the UNet based on the noisy x, noise condition, and the UNet kwargs
                pred = unet.forward(
                    x_noisy,
                    noise_cond,
                    **unet_kwargs
                ).detach()

                # Based on the noisy x at time t and the prediction, get the starting image - if the prediction objective is noise, the starting image is the noisy image, otherwise it is the prediction
                x_start = noise_scheduler.predict_start_from_noise(x_noisy, t = times, noise = pred) if pred_objective == 'noise' else pred

                # Set the new UNet kwargs to have a new self conditioning based on the new starting image
                unet_kwargs = {**unet_kwargs, 'self_cond': x_start}

        # Get prediction from the UNet based on the noisy x, noise condition, and the UNet kwargs
        pred = unet.forward(
            x_noisy,
            noise_cond,
            **unet_kwargs
        )

        # Prediction objective: what is the target? noise or image or v (?)?
        if pred_objective == 'noise':
            target = noise
        elif pred_objective == 'x_start':
            target = x_start
        elif pred_objective == 'v':
            # derivation detailed in Appendix D of Progressive Distillation paper
            # https://arxiv.org/abs/2202.00512
            # this makes distillation viable as well as solve an issue with color shifting in upresoluting unets, noted in imagen-video
            target = alpha * noise - sigma * x_start
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        # Compute the loss between the prediction and the target
        losses = self.loss_fn(pred, target, reduction = 'none')
        # Reduce the losses to the mean
        losses = reduce(losses, 'b ... -> b', 'mean')

        # Min snr loss reweighting (can help with stability of training)
        # Compute SNR
        snr = log_snr.exp()
        # Clone the SNR to maybe clip it
        maybe_clipped_snr = snr.clone()

        # If the minimum SNR gamma exists, clamp the clipped SNR to the minimum SNR gamma
        if exists(min_snr_gamma):
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        #  Compute loss weight
        if pred_objective == 'noise':
            # If objective is noiae, loss weight is the clipped SNR divided by the SNR
            loss_weight = maybe_clipped_snr / snr
        elif pred_objective == 'x_start':
            # If objective is x_start, loss weight is the clipped SNR
            loss_weight = maybe_clipped_snr
        elif pred_objective == 'v':
            # If objective is v, loss weight is the clipped SNR divided by the SNR plus 1
            loss_weight = maybe_clipped_snr / (snr + 1)

        # Multiply the losses by the loss weight
        losses = losses * loss_weight
        # Return the mean of the losses
        return losses.mean()

    @beartype
    def forward(
        self,                                       #
        images,                                     # rename to images or video         
        unet: Union[Unet, Unet3D, NullUnet, DistributedDataParallel] = None,  # Define the dtypes of UNet available
        texts: List[str] = None,                    # Text to condition on
        text_embeds = None,                         # Text embeddings
        text_masks = None,                          # Text masks
        unet_number = None,                         # UNet number
        cond_images = None,                         # Conditioning images
        **kwargs
    ):
        '''
        '''
        
        # If the model is working with videos and the images are 4D, add a frame dimension
        if self.is_video and images.ndim == 4:
            images = rearrange(images, 'b c h w -> b c 1 h w')
            kwargs.update(ignore_time = True)

        # Make sure that the images are square (height and width are equal)
        assert images.shape[-1] == images.shape[-2], f'the images you pass in must be a square, but received dimensions of {images.shape[2]}, {images.shape[-1]}'
        # Make sure that a specific UNet is specified to train
        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        
        # If no UNet is specified, train the first UNet
        unet_number = default(unet_number, 1)
        # If only_train_unet_number is specified make sure that the UNet number is equal to the only_train_unet_number
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you can only train on unet #{self.only_train_unet_number}'

        # Cast the images to float
        images = cast_uint8_images_to_float(images)
        # Cast the conditional images to float if they exist
        cond_images = maybe(cast_uint8_images_to_float)(cond_images)

        # Make sure that the images are floats
        assert images.dtype == torch.float or images.dtype == torch.half, f'images tensor needs to be floats but {images.dtype} dtype found instead'

        # Get the UNet index
        unet_index = unet_number - 1

        # Get the UNet if it exists, otherwise get the default UNet of the UNet index
        unet = default(unet, lambda: self.get_unet(unet_number))

        # Make sure that the UNet is not Null
        assert not isinstance(unet, NullUnet), 'null unet cannot and should not be trained'

        # Get the noise scheduler, minimum SNR gamma, prediction objective, target image size, random crop size, and previous image size
        noise_scheduler      = self.noise_schedulers[unet_index]
        min_snr_gamma        = self.min_snr_gamma[unet_index]
        pred_objective       = self.pred_objectives[unet_index]
        target_image_size    = self.image_sizes[unet_index]
        random_crop_size     = self.random_crop_sizes[unet_index]
        prev_image_size      = self.image_sizes[unet_index - 1] if unet_index > 0 else None

        # Get image shapes, device, and flag for video
        b, c, *_, h, w, device, is_video = *images.shape, images.device, images.ndim == 5

        # Make sure the number of channels on input images is equal to the number of channels on the model
        assert images.shape[1] == self.channels
        # Make sure that the image size is greater than or equal to the target image size
        assert h >= target_image_size and w >= target_image_size

        # Get the number of frames and dimensions if working with videos
        frames              = images.shape[2] if is_video else None
        all_frame_dims      = tuple(safe_get_tuple_index(el, 0) for el in calc_all_frame_dims(self.temporal_downsample_factor, frames))
        ignore_time         = kwargs.get('ignore_time', False) # ignore time for video conditioning

        # Get the target frame size, previous frame size and create a function to resize frames if working with videos
        target_frame_size   = all_frame_dims[unet_index] if is_video and not ignore_time else None
        prev_frame_size     = all_frame_dims[unet_index - 1] if is_video and not ignore_time and unet_index > 0 else None
        frames_to_resize_kwargs = lambda frames: dict(target_frames = frames) if exists(frames) else dict()

        # Sample random noise times from the noise scheduler
        times = noise_scheduler.sample_random_times(b, device = device)

        # Check if texts but text embeddings does not and if the model is not unconditional
        if exists(texts) and not exists(text_embeds) and not self.unconditional:
            # Make sure the texts are not empty (i.e. has a length)
            assert all([*map(len, texts)]), 'text cannot be empty'
            # Make sure that the number of texts is equal to the number of images
            assert len(texts) == len(images), 'number of text captions does not match up with the number of images given'

            # Disabling autocast to prevent OOM errors (autocast chooses the optimal dtype for each operation to improve performance)
            with autocast(enabled = False):
                # Encode the texts using the model's text encoder and return atention mask as well
                text_embeds, text_masks = self.encode_text(texts, return_attn_mask = True)

            # Set the text embeddings and text masks to the device
            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        # If the model is not unconditional
        if not self.unconditional:
            # Create text masks where the text embeddings are not equal to 0
            text_masks = default(text_masks, lambda: torch.any(text_embeds != 0., dim = -1))

        # Make sure that text embeddings exist if the model is conditiones on text
        assert not (self.condition_on_text and not exists(text_embeds)), 'text or text encodings must be passed into decoder if specified'
        # If the model is not conditioned on text, make sure that text embeddings do not exist
        assert not (not self.condition_on_text and exists(text_embeds)), 'decoder specified not to be conditioned on text, yet it is presented'

        # Make sure that, if text embeddings exist, the embedding dimension is equal to the model's text embedding dimension
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        # handle video frame conditioning
        # If the model is working with videos and the resize condition video frames flag is true
        if self.is_video and self.resize_cond_video_frames:
            # Get the temporal downsample scale for the current UNet
            downsample_scale = self.temporal_downsample_factor[unet_index]
            # Create a function to downsample the video frames by the downsample scale
            temporal_downsample_fn = partial(scale_video_time, downsample_scale = downsample_scale)
            # If the kwargs dict exists, transform the condition video frames and post condition video frames by the temporal downsample function
            kwargs = maybe_transform_dict_key(kwargs, 'cond_video_frames', temporal_downsample_fn)
            kwargs = maybe_transform_dict_key(kwargs, 'post_cond_video_frames', temporal_downsample_fn)

        # handle low resolution conditioning
        # Set low resolution conditioning image and low resolution augmentation times to none
        lowres_cond_img = lowres_aug_times = None

        # This two-step process simulates the conditions under which the model will be used at inference time.
        # Or simpler: We downsample, to lose information, but we still need the same image size to be passed in, so we upsample again - but the information is still lost.
        if exists(prev_image_size):
            # Resize the images to the previous image size (downsample)
            lowres_cond_img = self.resize_to(images, prev_image_size, **frames_to_resize_kwargs(prev_frame_size), clamp_range = self.input_image_range)
            # Resize the images to the target image size (upsample)
            lowres_cond_img = self.resize_to(lowres_cond_img, target_image_size, **frames_to_resize_kwargs(target_frame_size), clamp_range = self.input_image_range)

            # If working with individual random augmentation for each sample in batch
            if self.per_sample_random_aug_noise_level:
                # add random augmentation to the low resolution conditioning image at the low resolution augmentation times
                lowres_aug_times = self.lowres_noise_schedule.sample_random_times(b, device = device)
            else:
                # Else just repeat the same low resolution augmentation time for all samples in the batch
                lowres_aug_time = self.lowres_noise_schedule.sample_random_times(1, device = device)
                lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b = b)

        # Resize the images to the target image size
        images = self.resize_to(images, target_image_size, **frames_to_resize_kwargs(target_frame_size))

        # Return the losses calculated from the UNet given all the relevant arguments
        return self.p_losses(unet, 
                             images, 
                             times, 
                             text_embeds = text_embeds, 
                             text_mask = text_masks, 
                             cond_images = cond_images, 
                             noise_scheduler = noise_scheduler, 
                             lowres_cond_img = lowres_cond_img, 
                             lowres_aug_times = lowres_aug_times, 
                             pred_objective = pred_objective, 
                             min_snr_gamma = min_snr_gamma, 
                             random_crop_size = random_crop_size, 
                             **kwargs
                             )
'''
imagen_pytorch.py
-----------------
Purpose:
    This script appears to be focused on building and possibly training a neural network model using PyTorch.
    It imports a variety of utilities for tensor operations, distributed training, and more.

Key Components:
    - Torch and Torchvision: For building, training, and evaluating deep learning models.
    - beartype: For runtime type checking.
    - tqdm: Provides progress bars.
    - functools and contextlib: Utilities for higher-order functions and context management.
    - collections and pathlib: For data structures and filesystem path management.

Author:
    [lucidrains] On GitHub

Date:
    [October 3rd 2023] 

License:
    [License Information] (If applicable)
'''
import math # Standard Python module for mathematical operations.
import copy # Standard Python module for shallow and deep copy operations.
from beartype.typing import List, Union # Type hints from the beartype library.
#from beartype import beartype # Runtime type checking decorator from beartype.
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

#import kornia.augmentation as K # Computer vision augmentation library.

from einops import rearrange, repeat, reduce, pack, unpack # Rearrange tensor dimensions.
from einops.layers.torch import Rearrange, Reduce # Rearrange and reduce tensor dimensions.

#from CascadedDDPM.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME # T5 model and tokenizer utilities.

#from CascadedDDPM.imagen_video import Unet3D, resize_video_to, scale_video_time # 3D U-Net model and video utilities.

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


def cast_unit8_images_to_float(images):
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
        super().__init__()
        self.fns = nn.ModuleList(fns) # Creates a list of the functions to apply to the input tensor.

    def forward(self, x):
        '''
        Applies the module to the input tensor.
        
        Parameters:
        - x: Input tensor
        '''

        outputs = [fn(x) for fn in self.fns] # Applies each function to the input tensor.
        return sum(outputs) # Sums up the outputs of the functions.
    


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
    


def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )
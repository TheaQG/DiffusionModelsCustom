import torch
from torch import nn
from functools import reduce
from pathlib import Path

from configs import ImagenConfig, ElucidatedImagenConfig
from ema_pytorch import EMA

def exists(val):
    """
        Tests if val is not None
    """
    return val is not None
    


def safeget(dictionary, keys, default=None):
    """
        Retrieves a value from a nested dictionary (even if the keys do not exist(default is returned)), using 
        a dot-separated string of keys. If key is not found at any level, default is returned
    """
    return reduce(lambda d, key: d.get(key, default) if isinstance(d,dict) else default, keys.split("."), dictionary)


def load_imagen_from_checkpoint(
        checkpoint_path,
        load_weights = True,
        load_ema_if_available = False
    ):
    """
        
    """
    model_path = Path(checkpoint_path)
    full_model_path = str(model_path.resolve())
    assert model_path.exists(), f'Checkpoint not found at {full_model_path}'
    loaded = torch.load(str(model_path), map_location='cpu')
    
    imagen_params = safeget(loaded, 'imagen_params')
    imagen_type = safeget(loaded, 'imagen_type')

    if imagen_type == 'original':
        imagen_klass = ImagenConfig
    elif imagen_type == 'elucidated':
        imagen_klass = ElucidatedImagenConfig
    else:
        raise ValueError(f'Unknown imagen type {imagen_type} - you need to instantiate your imagen with configurations, using classes ImagenConfig or ElucidatedImagenConfig')
    
    assert exists(imagen_params) and exists(imagen_type), 'Imagen type and configuration not saved in this checkpoint'

    imagen = imagen_klass(**imagen_params).create()

    if not load_weights:
        return imagen
    has_ema = 'ema' in loaded
    should_load_ema = has_ema and load_ema_if_available

    imagen.load_state_dict(loaded['model'])

    if not should_load_ema:
        print('Loading non-EMA version of unets')
        return imagen
    
    ema_unets = nn.ModuleList([])
    for unet in imagen.unets:
        ema_unets.append(EMA(unet))

    ema_unets.load_state_dict(loaded['ema'])

    for unet, ema_unet in zip(imagen.unets, ema_unets):
        unet.load_state_dict(ema_unet.ema_model.state_dict())

    print('Loaded EMA version of unets')
    return imagen


    
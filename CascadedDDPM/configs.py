import json
from pydantic import BaseModel, validator
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any 
from enum import Enum

from imagen_pytorch.imagen_pytorch import Imagen, Unet, Unet3D, NullUnet
from imagen_pytorch.trainer import ImagenTrainer
from imagen_pytorch.elucidated_imagen import ElucidatedImagen
from imagen_pytorch.t5 import DEFAULT_T5_NAME, get_encoded_dim

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d 

def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]

def SingleOrList(inner_type):
    return Union[inner_type, ListOrTuple(inner_type)]


class NoiseSchedule(Enum):
    cosine = 'cosine'
    linear = 'linear'


class AllowExtraBaseModel(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True


class NullUnetConfig(BaseModel):
    is_null:            bool

    def create(self):
        return Unet(**self.dict())
    

class UnetConfig(AllowExtraBaseModel):
    dim:                int
    dim_mults:          ListOrTuple(int)
    text_embed_dim:     int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim:           int = None
    channels:           int = 3
    attn_dim_head:      int = 32
    attn_heads:         int = 16

    def create(self):
        return Unet(**self.dict())

class Unet3DConfig(AllowExtraBaseModel):
    dim:                int
    dim_mults:          ListOrTuple(int)
    text_embed_dim:     int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim:           int = None
    channels:           int = 3
    attn_dim_head:      int = 32
    attn_heads:         int = 16

    def create(self):
        return Unet3D(**self.dict())
    

class ImagenConfig(AllowExtraBaseModel):
    unets:              ListOrTuple(Union[UnetConfig, Unet3DConfig, NullUnetConfig])
    image_sizes:        ListOrTuple(int)
    video:              bool = False
    timesteps:          SingleOrList(int) = 1000
    noise_schedules:    SingleOrList(NoiseSchedule) = 'cosine'
    text_encoder_name:  str = DEFAULT_T5_NAME
    channels:           int = 3
    loss_type:          str = '12'
    cond_drop_prob:     float = 0.5

    @validator('image_sizes')
    def check_image_sizes(cls, image_sizes, values):
        unets = values.get('unets')
        if len(image_sizes) != len(unets):
            raise ValueError(f'Image sizes length {len(image_sizes)} must be equivalent to the number of unets {len(unets)}')
        return image_sizes
    
    def create(self):
        decoder_kwargs = self.dict()
        unets_kwargs = decoder_kwargs.pop('unets')
        is_video = decoder_kwargs.pop('video', False)

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = Unet3D
            else:
                unet_klass = Unet

            unets.append(unet_klass(**unet_kwargs))

        imagen = Imagen(unets, **decoder_kwargs)

        imagen._config = self.dict().copy()

        return imagen

    

        
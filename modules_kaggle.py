import os, tqdm, random, torch
import numpy as np
import torch.nn as nn
from PIL import Image
from multiprocessing import Manager as SharedMemoryManager
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Union, Iterable, Tuple


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim_size, n:int = 10000):
        assert dim_size % 2 == 0, 'dim_size must be even'

        super(SinusoidalEmbedding, self).__init__()

        self.dim_size = dim_size
        self.n = n

    def forward(self, x:torch.Tensor):
        N = len(x)
        output = torch.zeros(size = (N, self.dim_size)).to(x.device)

        for idx in range(0,N):
            for i in range(0, self.dim_size//2):
                emb = x[idx] / (self.n ** (2*i / self.dim_size))
                output[idx, 2*i] = torch.sin(emb)
                output[idx, 2*i+1] = torch.cos(emb)

        return output
    

class ImageSelfAttention(nn.Module):
    def __init__(self, input_channels:int, n_heads:int):
        super(ImageSelfAttention, self).__init__()
        
        self.input_channels = input_channels
        self.n_heads = n_heads
        self.layernorm = nn.LayerNorm(self.input_channels)
        self.attention = nn.MultiheadAttention(self.input_channels, self.n_heads, batch_first=True)
        
    def forward(self, x:torch.Tensor):
        # shape of x: (N, C, H, W)
        _, C, H, W = x.shape
        x = x.reshape(-1, C, H*W).permute(0, 2, 1)
        normalised_x = self.layernorm(x)
        attn_val, _ = self.attention(normalised_x, normalised_x, normalised_x)
        attn_val = attn_val + x
        attn_val = attn_val.permute(0, 2, 1).reshape(-1, C, H, W)
        return attn_val
    

class Encoder(ResNet):
    def __init__(
        self, input_channels:int, time_embedding:int, 
        block=BasicBlock, block_layers:list=[2, 2, 2, 2], n_heads:int=4):
      
        self.block = block
        self.block_layers = block_layers
        self.time_embedding = time_embedding
        self.input_channels = input_channels
        self.n_heads = n_heads
        
        super(Encoder, self).__init__(self.block, self.block_layers)
        
        #time embedding layer
        self.sinusiodal_embedding = SinusoidalEmbedding(self.time_embedding)
        
        fmap_channels = [64, 64, 128, 256, 512]
        #layers to project time embeddings unto feature maps
        self.time_projection_layers = self.make_time_projections(fmap_channels)
        #attention layers for each feature map
        self.attention_layers = self.make_attention_layers(fmap_channels)
        
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False)
        
        self.conv2 = nn.Conv2d(
            64, 64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3),
            bias=False)

        #delete unwanted layers
        del self.maxpool, self.fc, self.avgpool
        
        
    def forward(self, x:torch.Tensor, t:torch.Tensor):
        #embed time positions
        t = self.sinusiodal_embedding(t)
        
        #prepare fmap2
        fmap1 = self.conv1(x)
        t_emb = self.time_projection_layers[0](t)
        fmap1 = fmap1 + t_emb[:, :, None, None]
        fmap1 = self.attention_layers[0](fmap1)
        
        x = self.conv2(fmap1)
        x = self.bn1(x)
        x = self.relu(x)
        
        #prepare fmap2
        fmap2 = self.layer1(x)
        t_emb = self.time_projection_layers[1](t)
        fmap2 = fmap2 + t_emb[:, :, None, None]
        fmap2 = self.attention_layers[1](fmap2)
        
        #prepare fmap3
        fmap3 = self.layer2(fmap2)
        t_emb = self.time_projection_layers[2](t)
        fmap3 = fmap3 + t_emb[:, :, None, None]
        fmap3 = self.attention_layers[2](fmap3)
        
        #prepare fmap4
        fmap4 = self.layer3(fmap3)
        t_emb = self.time_projection_layers[3](t)
        fmap4 = fmap4 + t_emb[:, :, None, None]
        fmap4 = self.attention_layers[3](fmap4)
        
        #prepare fmap4
        fmap5 = self.layer4(fmap4)
        t_emb = self.time_projection_layers[4](t)
        fmap5 = fmap5 + t_emb[:, :, None, None]
        fmap5 = self.attention_layers[4](fmap5)
        
        return fmap1, fmap2, fmap3, fmap4, fmap5
    
    
    def make_time_projections(self, fmap_channels:Iterable[int]):
        layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, ch)
            ) for ch in fmap_channels ])
        
        return layers
    
    def make_attention_layers(self, fmap_channels:Iterable[int]):
        layers = nn.ModuleList([
            ImageSelfAttention(ch, self.n_heads) for ch in fmap_channels
        ])
        
        return layers
    



class DecoderBlock(nn.Module):
    def __init__(
        self, input_channels:int, output_channels:int, 
        time_embedding:int, upsample_scale:int=2, activation:nn.Module=nn.ReLU,
        compute_attn:bool=True, n_heads:int=4):
        super(DecoderBlock, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upsample_scale = upsample_scale
        self.time_embedding = time_embedding
        self.compute_attn = compute_attn
        self.n_heads = n_heads
        
        #attention layer
        if self.compute_attn:
            self.attention = ImageSelfAttention(self.output_channels, self.n_heads)
        else:self.attention = nn.Identity()
        
        #time embedding layer
        self.sinusiodal_embedding = SinusoidalEmbedding(self.time_embedding)
        
        #time embedding projection layer
        self.time_projection_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, self.output_channels)
            )

        self.transpose = nn.ConvTranspose2d(
            self.input_channels, self.input_channels, 
            kernel_size=self.upsample_scale, stride=self.upsample_scale)
        
        self.instance_norm1 = nn.InstanceNorm2d(self.transpose.in_channels)

        self.conv = nn.Conv2d(
            self.transpose.out_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        
        self.instance_norm2 = nn.InstanceNorm2d(self.conv.out_channels)
        
        self.activation = activation()

    
    def forward(self, fmap:torch.Tensor, prev_fmap:Optional[torch.Tensor]=None, t:Optional[torch.Tensor]=None):
        output = self.transpose(fmap)
        output = self.instance_norm1(output)
        output = self.conv(output)
        output = self.instance_norm2(output)
        
        #apply residual connection with previous feature map
        if torch.is_tensor(prev_fmap):
            assert (prev_fmap.shape == output.shape), 'feature maps must be of same shape'
            output = output + prev_fmap
            
        #apply timestep embedding
        if torch.is_tensor(t):
            t = self.sinusiodal_embedding(t)
            t_emb = self.time_projection_layer(t)
            output = output + t_emb[:, :, None, None]
            
            output = self.attention(output)
            
        output = self.activation(output)
        return output
    


class Decoder(nn.Module):
    def __init__(
        self, last_fmap_channels:int, output_channels:int, 
        time_embedding:int, first_fmap_channels:int=64, n_heads:int=4):
        super(Decoder, self).__init__()
        
        self.last_fmap_channels = last_fmap_channels
        self.output_channels = output_channels
        self.time_embedding = time_embedding
        self.first_fmap_channels = first_fmap_channels
        self.n_heads = n_heads

        self.residual_layers = self.make_layers()

        self.final_layer = DecoderBlock(
            self.residual_layers[-1].input_channels, self.output_channels,
            time_embedding=self.time_embedding, activation=nn.Identity, 
            compute_attn=False, n_heads=self.n_heads)

        #set final layer second instance norm to identity
        self.final_layer.instance_norm2 = nn.Identity()


    def forward(self, *fmaps, t:Optional[torch.Tensor]=None):
        #fmaps(reversed): fmap5, fmap4, fmap3, fmap2, fmap1
        fmaps = [fmap for fmap in reversed(fmaps)]
        ouptut = None
        for idx, m in enumerate(self.residual_layers):
            if idx == 0:
                output = m(fmaps[idx], fmaps[idx+1], t)
                continue
            output = m(output, fmaps[idx+1], t)
        
        # no previous fmap is passed to the final decoder block
        # and no attention is computed
        output = self.final_layer(output)
        return output

      
    def make_layers(self, n:int=4):
        layers = []
        for i in range(n):
            if i == 0: in_ch = self.last_fmap_channels
            else: in_ch = layers[i-1].output_channels

            out_ch = in_ch // 2 if i != (n-1) else self.first_fmap_channels
            layer = DecoderBlock(
                in_ch, out_ch, 
                time_embedding=self.time_embedding,
                compute_attn=True, n_heads=self.n_heads)
            
            layers.append(layer)

        layers = nn.ModuleList(layers)
        return layers
    



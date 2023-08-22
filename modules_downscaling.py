import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm

class UNet(nn.module):
    def __init__(self, c_in=3, c_out=1, time_dim=256, device='cpu'):

        super().__init__()
        self.device = device
        self.time_dim = time_dim

        '''
            First, two convolution layers (.inc) to reduce the number of channels
            Then, 3 downsampling blocks (.down1, .down2, .down3) followed by self-attention
            blocks (.sa1, .sa2, .sa3)

            Argument for downsampling blocks: (c_in, c_out)
            Argument for self-attention blocks: (c_in, current image resolution) 

            Each downsampling block reduces im size by 2 (64 --> 32 --> 16 --> 8)
        '''
        self.inc = DoubleConv(c_in, 64) # Start with 2 convolutional layers
        self.down1 = Down(64, 128) # 1st downsampling block
        self.sa1 = SelfAttention(128, 32) # 1st self-attention block
        self.down2 = Down(128, 256) # 2nd downsampling block
        self.sa2 = SelfAttention(256, 16) # 2nd self-attention block
        self.down3 = Down(256, 256) # 3rd downsampling block
        self.sa3 = SelfAttention(256, 8) # 3rd self-attention block


        '''
            Bottleneck block. Just a number of convolutional layers
        '''
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)


        '''
            Basically the reverse of the downsampling blocks.
            Three upsampling blocks (.up1, .up2, .up3) followed by self-attention blocks
            (.sa4, .sa5, .sa6)

            Finally we project back to the output channel dimension w a convolutional layer
        '''
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        '''
            Sinusoidal positional encoding (sinusoidal embedding).
            To make it easier for the model to learn the time dimension
            Argument for sinusoidal embedding??
        '''
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels,2, device=self.device).float() / channels)) 

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq) # Repeat t to match the number of channels
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq) # Repeat t to match the number of channels
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1) # Concatenate along the channel dimension

        return pos_enc


    def forward(self, x, t):
        ''' 
            Forward pass of the UNet model.
            Sinusoidal embedding is applied to the time dimension before the first convolutional layer.
            Skip-connections are used to connect the downsampling blocks to the upsampling blocks.
        '''
        t = t.unsqueeze(-1).type(torch.float32) # Add a dimension to t
        t = self.pos_encoding(t, self.time_dim) # Sinusoidal embedding

        x1 = self.inc1(x) # First convolutional layer
        x2 = self.down1(x1, t) # First downsampling block
        x2 = self.sa1(x2) # First self-attention block
        x3 = self.down2(x2, t) # Second downsampling block
        x3 = self.sa2(x3) # Second self-attention block
        x4 = self.down3(x3, t) # Third downsampling block
        x4 = self.sa3(x4) # Third self-attention block

        x4 = self.bot1(x4) # First bottleneck block
        x4 = self.bot2(x4) # Second bottleneck block
        x4 = self.bot3(x4) # Third bottleneck block

        x = self.up1(x4, x3, t) # First upsampling block (takes skip-connection x3 as input as well)
        x = self.sa4(x) # Fourth self-attention block
        x = self.up2(x, x2, t) # Second upsampling block (takes skip-connection x2 as input as well)
        x = self.sa5(x) # Fifth self-attention block
        x = self.up3(x, x1, t) # Third upsampling block (takes skip-connection x1 as input as well)
        x = self.sa6(x) # Sixth self-attention block

        output = self.outc(x) # Output layer

        return output


class DoubleConv(nn.Module):
    '''
        Normal convolution block: Convolutional layer --> Group normalization --> GELU activation
        Possibility to add a residual connection (residual=True)
        More info on GroupNorm: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        Mid_channels is the number of channels in the middle convolutional layer
        If not specified, mid_channels = out_channels
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()

        self.residual = residual # Residual connection

        if not mid_channels:
            mid_channels = out_channels # If not specified, mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # Convolutional layer
            nn.GroupNorm(1, mid_channels), # Group normalization
            nn.GELU(), # GELU activation
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), # Convolutional layer
            nn.GroupNorm(1, out_channels), # Group normalization
        )
    
    def forward(self, x):
        if self.residual:
            return F.GELU(x + self.double_conv(x)) # Residual connection
        else:
            return self.double_conv(x)
        

class Down(nn.Module):
    '''
        Maxpooling to half the size of the input, followed by two double convolutional layer.
        Here is the embedding layer (encoding time to certain dimension)
        From time embedding to output dimension
    '''
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # Max pooling
            DoubleConv(in_channels, in_channels, residual=True), # Double convolutional layer w. residual connection
            DoubleConv(in_channels, out_channels), # Double convolutional layer
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(), # SiLU activation
            nn.Linear(emb_dim, out_channels), # Linear layer
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x) # Max pooling + double convolutional layer
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # Embedding layer (repeat to match the spatial dimensions of x)

        return x + emb # Add the embedding to the output of the max pooling + double convolutional layer
    

class Up(nn.Module):
    '''
        Upsampling to double the size of the input, followed by two double convolutional layer.
        Takes also the skip-connection that comes from the encoder.
        Adds the timeembedding again 
    '''
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Upsampling
        self.conv = nn.Sequential( 
            DoubleConv(in_channels, in_channels, residual=True), # Double convolutional layer w. residual connection
            DoubleConv(in_channels, out_channels, in_channels // 2), # Double convolutional layer
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(), # SiLU activation
            nn.Linear(emb_dim, out_channels), # Linear layer
        )

    def forward(self, x, skip_x, t):
        x = self.up(x) # Upsampling
        x = torch.cat([skip_x, x], dim=1) # Concatenate skip-connection and upsampled output
        x = self.conv(x) # Double convolutional layer
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # Embedding layer (repeat to match the spatial dimensions of x)
        return x + emb # Add the embedding to the output of the double convolutional layer



class SelfAttention(nn.Module):
    '''
        Completely normal attention block as in Transformers

    '''
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()

        self.channels = channels # Number of channels
        self.size = size # Size of the image
        self.mha = nn.MultiHeadAttention(channels, 4, batch_first=True) # Multi-head attention layer (4 heads) 
        self.ln = nn.LayerNorm([channels]) # Layer normalization layer (normalizes the output of the multi-head attention layer)
        self.ff_self = nn.Sequential( # Feed-forward layer
            nn.LayerNorm([channels]), 
            nn.Linear(channels, channels),
            nn.GELU(), 
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2) # Reshape the input to match the input of the multi-head attention layer
        x_ln = self.ln(x) # Layer normalization
        attention_value, _ = self.mha(x_ln, x_ln, x_ln) # Multi-head attention 
        attention_value = attention_value + x # Add the input to the output of the multi-head attention layer (skip conncetion)
        attention_value = self.ff_self(attention_value) + attention_value # Feed-forward layer 

        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size) # Reshape the output to match the input of the next layer

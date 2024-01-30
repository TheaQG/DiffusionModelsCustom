'''
    Script containing neural network modules for DDPM_DANRA_Downscaling.
    The encoder and decoder modules are used in a UNET for downscaling in the DDPM.
    The following modules are defined:
        - SinusoidalEmbedding: sinusoidal embedding module
        - ImageSelfAttention: image self-attention module
        - Encoder: encoder module
        - DecoderBlock: decoder block module
        - Decoder: decoder module
        
'''
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Iterable

class SinusoidalEmbedding(nn.Module):
    '''
        Class for sinusoidal embedding.
        The sinusoidal embedding is used to embed the time information into the data.
        
    '''
    def __init__(self, dim_size, n:int = 10000):
        '''
            Initialize the class.
            Input:
                - dim_size: size of the embedding
                - n: number of sinusoids
            
        '''

        # Check if dim_size is even
        assert dim_size % 2 == 0, 'dim_size must be even'

        # Initialize the class
        super(SinusoidalEmbedding, self).__init__()

        # Set the class variables
        self.dim_size = dim_size
        self.n = n

    def forward(self, x:torch.Tensor):
        '''
            Forward function for the class. The sinusoidal embedding is applied to the input.
            Input:
                - x: input tensor
        '''
        # Get the length of the input
        N = len(x)

        # Initialize the output tensor
        output = torch.zeros(size = (N, self.dim_size)).to(x.device)

        # Loop over the input
        for idx in range(0,N):
            for i in range(0, self.dim_size//2):
                emb = x[idx] / (self.n ** (2*i / self.dim_size))
                # Calculate the sinusoidal embedding
                output[idx, 2*i] = torch.sin(emb)
                # Calculate the cosinusoidal embedding
                output[idx, 2*i+1] = torch.cos(emb)

        return output



class ImageSelfAttention(nn.Module):
    ''' 
        Class for image self-attention. Self-attention is a mechanism that allows the model to focus on more important features.
        Focus on one thing and ignore other things that seem irrelevant at the moment.
    '''
    def __init__(self, input_channels:int, n_heads:int):
        '''
            Initialize the class.
            Input:
                - input_channels: number of input channels
                - n_heads: number of heads (how many different parts of the input are attended to)
        '''
        # Initialize the class
        super(ImageSelfAttention, self).__init__()
        
        # Set the class variables
        self.input_channels = input_channels
        self.n_heads = n_heads
        # Layer normalization layer, for normalizing the input
        self.layernorm = nn.LayerNorm(self.input_channels)
        # Multi-head attention layer, for calculating the attention
        self.attention = nn.MultiheadAttention(self.input_channels, self.n_heads, batch_first=True)
        
    def forward(self, x:torch.Tensor):
        '''
            Forward function for the class. The self-attention is applied to the input x.
            Self-attention is calculated by calculating the dot product of the input with itself.
        '''

        # shape of x: (N, C, H, W), (N samples, C channels, height, width)
        _, C, H, W = x.shape

        # Reshape the input to (N, C, H*W) and permute to (N, H*W, C)
        x = x.reshape(-1, C, H*W).permute(0, 2, 1)
        # Normalize the input
        normalised_x = self.layernorm(x)
        # Calculate the attention value and attention weights 
        attn_val, _ = self.attention(normalised_x, normalised_x, normalised_x)
        # Add the attention value to the input
        attn_val = attn_val + x
        # Reshape the attention value to (N, C, H, W)
        attn_val = attn_val.permute(0, 2, 1).reshape(-1, C, H, W)
        return attn_val






class Encoder(ResNet):
    '''
        Class for the encoder. The encoder is used to encode the input data.
        The encoder is a ResNet with self-attention layers, and will be part of a UNET used for downscaling in the DDPM.
        The encoder consists of five feature maps, one for each layer of the ResNet.
        The encoder works as a downsample block, and will be used to downsample the input.
    '''
    def __init__(self,
                 input_channels:int,
                 time_embedding:int, 
                 block=BasicBlock,
                 block_layers:list=[2, 2, 2, 2],
                 n_heads:int=4,
                 num_classes:int=None,
                 lsm_tensor=None,
                 topo_tensor=None,
                 cond_on_img=False,
                 cond_img_dim = None
                 ):
        '''
            Initialize the class. 
            Input:
                - input_channels: number of input channels
                - time_embedding: size of the time embedding
                - block: block to use for the ResNet
                - block_layers: list containing the number of blocks for each layer (default: [2, 2, 2, 2], 4 layers with 2 blocks each)
                - n_heads: number of heads for the self-attention layers (default: 4, meaning 4 heads for each self-attention layer)
        '''
        # Initialize the class
        self.block = block
        self.block_layers = block_layers
        self.time_embedding = time_embedding
        self.input_channels = input_channels
        self.n_heads = n_heads
        self.num_classes = num_classes
        
        # Initialize the ResNet with the given block and block_layers
        super(Encoder, self).__init__(self.block, self.block_layers)

        # Register the lsm and elevation tensors as buffers (i.e. they are not trained) and add a channel for each of them
        if lsm_tensor is not None:
            self.register_buffer('lsm', lsm_tensor)
            self.input_channels += 1
        if topo_tensor is not None:
            self.register_buffer('elevation', topo_tensor)
            self.input_channels += 1
        if cond_on_img:
            self.input_channels += cond_img_dim[0]
        
        # Initialize the sinusoidal time embedding layer with the given time_embedding
        self.sinusiodal_embedding = SinusoidalEmbedding(self.time_embedding)
        
        # Set the channels for the feature maps (five feature maps, one for each layer, with 64, 64, 128, 256, 512 channels)
        fmap_channels = [64, 64, 128, 256, 512]

        # Set the time projection layers, for projecting the time embedding onto the feature maps
        self.time_projection_layers = self.make_time_projections(fmap_channels)
        # Set the attention layers, for calculating the attention for each feature map
        self.attention_layers = self.make_attention_layers(fmap_channels)
        
        # Set the first convolutional layer, with N input channels(=input_channels) and 64 output channels
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(8, 8), # Previous kernelsize (7,7)
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False)
        
        # Set the second convolutional layer, with 64 input channels and 64 output channels
        self.conv2 = nn.Conv2d(
            64, 64, 
            kernel_size=(8, 8), # Previous kernelsize (7,7)
            stride=(2, 2), 
            padding=(3, 3),
            bias=False)

        # If conditional, set the label embedding layer from the number of classes to the time embedding size
        if num_classes is not None:
            
            self.label_emb = nn.Embedding(num_classes, time_embedding)

        #delete unwanted layers, i.e. maxpool(=self.maxpool), fully connected layer(=self.fc) and average pooling(=self.avgpool
        del self.maxpool, self.fc, self.avgpool

        
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            1000
            ** (torch.arange(0, channels, 2).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, 
                x:torch.Tensor, 
                t:torch.Tensor, 
                y:Optional[torch.Tensor]=None, 
                cond_img:Optional[torch.Tensor]=None, 
                lsm_cond:Optional[torch.Tensor]=None, 
                topo_cond:Optional[torch.Tensor]=None
                ):
        '''
            Forward function for the class. The input x and time embedding t are used to calculate the output.
            The output is the encoded input x.
            Input:
                - x: input tensor
                - t: time embedding tensor
        '''
        if hasattr(self, 'lsm'):
            lsm_cond = lsm_cond#.unsqueeze(0).unsqueeze(0)
            x = torch.cat([x, lsm_cond], dim=1)
            #x = torch.cat([x, self.lsm.repeat(x.size(0), 1, 1, 1)], dim=1)
        if hasattr(self, 'elevation'):
            topo_cond = topo_cond#.unsqueeze(0).unsqueeze(0)
            x = torch.cat([x, topo_cond], dim=1)
            #x = torch.cat([x, self.elevation.repeat(x.size(0), 1, 1, 1)], dim=1)

        if cond_img is not None:
            x = torch.cat((x, cond_img), dim=1)
            #x = x.to(torch.double)
            #print('\n Conditional image added to input with dtype: ', x.dtype, '\n')

        # Embed the time positions
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_embedding)#self.num_classes)

        #t = self.sinusiodal_embedding(t)
        # Add the label embedding to the time embedding
        if y is not None:
            # print('Time embedding size:')
            # print(t.shape)  
            # print('Label size:')
            # print(y.shape)
            # print('Label embedding size:')
            # print(self.label_emb(y).shape)

            t += self.label_emb(y)
        #print('\n Time embedding type: ', t.dtype, '\n')
        # Prepare fmap1, the first feature map, by applying the first convolutional layer to the input x
        
        fmap1 = self.conv1(x)
        # Project the time embedding onto fmap1
        t_emb = self.time_projection_layers[0](t)
        # Add the projected time embedding to fmap1
        fmap1 = fmap1 + t_emb[:, :, None, None]
        # Calculate the attention for fmap1
        fmap1 = self.attention_layers[0](fmap1)
        
        # Prepare fmap2, the second feature map, by applying the second convolutional layer to fmap1
        x = self.conv2(fmap1)
        # Normalize fmap2 with batch normalization
        x = self.bn1(x)
        # Apply the ReLU activation function to fmap2
        x = self.relu(x)
        
        # Prepare fmap2, the second feature map, by applying the first layer of blocks to fmap2
        fmap2 = self.layer1(x)
        # Project the time embedding onto fmap2 
        t_emb = self.time_projection_layers[1](t)
        # Add the projected time embedding to fmap2
        fmap2 = fmap2 + t_emb[:, :, None, None]
        # Calculate the attention for fmap2
        fmap2 = self.attention_layers[1](fmap2)
        
        # Prepare fmap3, the third feature map, by applying the second layer of blocks to fmap2
        fmap3 = self.layer2(fmap2)
        # Project the time embedding onto fmap3
        t_emb = self.time_projection_layers[2](t)
        # Add the projected time embedding to fmap3
        fmap3 = fmap3 + t_emb[:, :, None, None]
        # Calculate the attention for fmap3
        fmap3 = self.attention_layers[2](fmap3)
        
        # Prepare fmap4, the fourth feature map, by applying the third layer of blocks to fmap3
        fmap4 = self.layer3(fmap3)
        # Project the time embedding onto fmap4
        t_emb = self.time_projection_layers[3](t)
        # Add the projected time embedding to fmap4
        fmap4 = fmap4 + t_emb[:, :, None, None]
        # Calculate the attention for fmap4
        fmap4 = self.attention_layers[3](fmap4)
        
        # Prepare fmap5, the fifth feature map, by applying the fourth layer of blocks to fmap4
        fmap5 = self.layer4(fmap4)
        # Project the time embedding onto fmap5
        t_emb = self.time_projection_layers[4](t)
        # Add the projected time embedding to fmap5
        fmap5 = fmap5 + t_emb[:, :, None, None]
        # Calculate the attention for fmap5
        fmap5 = self.attention_layers[4](fmap5)
        
        # Return the feature maps
        return fmap1, fmap2, fmap3, fmap4, fmap5
    
    
    def make_time_projections(self, fmap_channels:Iterable[int]):
        '''
            Function for making the time projection layers. The time projection layers are used to project the time embedding onto the feature maps.
            Input:
                - fmap_channels: list containing the number of channels for each feature map
        '''
        # Initialize the time projection layers consisting of a SiLU activation function and a linear layer. 
        # The SiLU activation function is used to introduce non-linearity. One time projection layer is used for each feature map.
        # The number of input channels for each time projection layer is the size of the time embedding, and the number of output channels is the number of channels for the corresponding feature map.
        # Only the first time projection layer has a different number of input channels, namely the number of input channels for the first convolutional layer.
        layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, ch)
            ) for ch in fmap_channels ])
        
        return layers
    
    def make_attention_layers(self, fmap_channels:Iterable[int]):
        '''
            Function for making the attention layers. The attention layers are used to calculate the attention for each feature map.
            Input:
                - fmap_channels: list containing the number of channels for each feature map
        '''
        # Initialize the attention layers. One attention layer is used for each feature map.
        layers = nn.ModuleList([
            ImageSelfAttention(ch, self.n_heads) for ch in fmap_channels
        ])
        
        return layers
    



class DecoderBlock(nn.Module):
    '''
        Class for the decoder block. The decoder block is used to decode the encoded input.
        Part of a UNET used for downscaling in the DDPM. The decoder block consists of a transposed convolutional layer, a convolutional layer, and a self-attention layer.
        The decoder block works as an upsample block, and will be used to upsample the input.
    '''
    def __init__(
            self,
            input_channels:int,
            output_channels:int,
            time_embedding:int,
            upsample_scale:int=2,
            activation:nn.Module=nn.ReLU,
            compute_attn:bool=True,
            n_heads:int=4
            ):
        '''
            Initialize the class.
            Input:
                - input_channels: number of input channels
                - output_channels: number of output channels
                - time_embedding: size of the time embedding
                - upsample_scale: scale factor for the transposed convolutional layer (default: 2, meaning the output will be twice the size of the input)
                - activation: activation function to use (default: ReLU)
                - compute_attn: boolean indicating whether to compute the attention (default: True)
                - n_heads: number of heads for the self-attention layer (default: 4, meaning 4 heads for the self-attention layer)
        '''

        # Initialize the class
        super(DecoderBlock, self).__init__()
        

        # Initialize the class variables
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upsample_scale = upsample_scale
        self.time_embedding = time_embedding
        self.compute_attn = compute_attn
        self.n_heads = n_heads
        
        # Initialize the attention layer, if compute_attn is True
        if self.compute_attn:
            # Initialize the attention layer
            self.attention = ImageSelfAttention(self.output_channels, self.n_heads)
        else:
            # Initialize the identity layer as the attention layer
            self.attention = nn.Identity()
        
        # Initialize the sinusoidal time embedding layer with the given time_embedding
        self.sinusiodal_embedding = SinusoidalEmbedding(self.time_embedding)
        
        # Initialize the time projection layer, for projecting the time embedding onto the feature maps. SiLU activation function and linear layer.
        self.time_projection_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, self.output_channels)
            )

        # Initialize the transposed convolutional layer. 
        self.transpose = nn.ConvTranspose2d(
            self.input_channels, self.input_channels, 
            kernel_size=self.upsample_scale, stride=self.upsample_scale)
        
        # Define the instance normalization layer, for normalizing the input
        self.instance_norm1 = nn.InstanceNorm2d(self.transpose.in_channels)

        # Define the convolutional layer
        self.conv = nn.Conv2d(
            self.transpose.out_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        
        # Define second instance normalization layer, for normalizing the input
        self.instance_norm2 = nn.InstanceNorm2d(self.conv.out_channels)
        
        # Define the activation function
        self.activation = activation()

    
    def forward(self,
                fmap:torch.Tensor,
                prev_fmap:Optional[torch.Tensor]=None,
                t:Optional[torch.Tensor]=None
                ):
        '''
            Forward function for the class. The input fmap, previous feature map prev_fmap, and time embedding t are used to calculate the output.
            The output is the decoded input fmap.
        '''
        # Prepare the input fmap by applying a transposed convolutional, instance normalization, convolutional, and second instance norm layers
        output = self.transpose(fmap)
        output = self.instance_norm1(output)
        output = self.conv(output)
        output = self.instance_norm2(output)
        
        # Apply residual connection with previous feature map. If prev_fmap is a tensor, the feature maps must be of the same shape.
        if torch.is_tensor(prev_fmap):
            assert (prev_fmap.shape == output.shape), 'feature maps must be of same shape'
            # Add the previous feature map to the output
            output = output + prev_fmap
            
        # Apply timestep embedding if t is a tensor
        if torch.is_tensor(t):
            # Embed the time positions
            t = self.sinusiodal_embedding(t)
            # Project the time embedding onto the feature maps
            t_emb = self.time_projection_layer(t)
            # Add the projected time embedding to the output
            output = output + t_emb[:, :, None, None]
            
            # Calculate the attention for the output
            output = self.attention(output)
        
        # Apply the activation function to the output
        output = self.activation(output)
        return output
    



class Decoder(nn.Module):
    '''
        Class for the decoder. The decoder is used to decode the encoded input.
        The decoder is a UNET with self-attention layers, and will be part of a UNET used for downscaling in the DDPM.
        The decoder consists of five feature maps, one for each layer of the UNET.
        The decoder works as an upsample block, and will be used to upsample the input.
    '''
    def __init__(self,
                 last_fmap_channels:int,
                 output_channels:int,
                 time_embedding:int,
                 first_fmap_channels:int=64,
                 n_heads:int=4
                 ):
        '''
            Initialize the class. 
            Input:
                - last_fmap_channels: number of channels for the last feature map
                - output_channels: number of output channels
                - time_embedding: size of the time embedding
                - first_fmap_channels: number of channels for the first feature map (default: 64)
                - n_heads: number of heads for the self-attention layers (default: 4, meaning 4 heads for each self-attention layer)
        '''

        # Initialize the class
        super(Decoder, self).__init__()
        
        # Initialize the class variables
        self.last_fmap_channels = last_fmap_channels
        self.output_channels = output_channels
        self.time_embedding = time_embedding
        self.first_fmap_channels = first_fmap_channels
        self.n_heads = n_heads

        # Initialize the residual layers (four residual layers)
        self.residual_layers = self.make_layers()

        # Initialize the final layer, a decoder block without previous feature map and without attention
        self.final_layer = DecoderBlock(
            self.residual_layers[-1].input_channels, self.output_channels,
            time_embedding=self.time_embedding, activation=nn.Identity, 
            compute_attn=False, n_heads=self.n_heads)

        # Set final layer second instance norm to identity as the final layer does not have a previous feature map
        self.final_layer.instance_norm2 = nn.Identity()


    def forward(self, *fmaps, t:Optional[torch.Tensor]=None):
        '''
            Forward function for the class.
            Input:
                - fmaps: feature maps
                - t: time embedding tensor
        '''
        # Reverse the feature maps in a list, fmaps(reversed): fmap5, fmap4, fmap3, fmap2, fmap1
        fmaps = [fmap for fmap in reversed(fmaps)]

        output = None

        # Loop over the residual layers
        for idx, m in enumerate(self.residual_layers):
            if idx == 0:
                # If idx is 0, the first residual layer is used.
                output = m(fmaps[idx], fmaps[idx+1], t)
                continue
            # If idx is not 0, the other residual layers are used.
            output = m(output, fmaps[idx+1], t)
        
        # No previous fmap is passed to the final decoder block
        # and no attention is computed
        output = self.final_layer(output)
        return output

      
    def make_layers(self, n:int=4):
        '''
            Function for making the residual layers. 
            Input:
                - n: number of residual layers (default: 4)
        '''
        # Initialize the residual layers
        layers = []

        # Loop over the number of residual layers
        for i in range(n):
            # If i is 0, the first residual layer is used.
            if i == 0: in_ch = self.last_fmap_channels
            # If i is not 0, the other residual layers are used.
            else: in_ch = layers[i-1].output_channels

            # Set the number of output channels for the residual layer
            out_ch = in_ch // 2 if i != (n-1) else self.first_fmap_channels

            # Initialize the residual layer as a decoder block
            layer = DecoderBlock(
                in_ch, out_ch, 
                time_embedding=self.time_embedding,
                compute_attn=True, n_heads=self.n_heads)
            
            # Add the residual layer to the list of residual layers
            layers.append(layer)

        # Return the residual layers as a ModuleList
        layers = nn.ModuleList(layers)
        return layers

class DiffusionNet(nn.Module):
    '''
        Class for the diffusion net. The diffusion net is used to encode and decode the input.
        The diffusion net is a UNET with self-attention layers, and will be used for downscaling in the DDPM.
    '''
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder,
                 lsm_tensor=None,
                 topo_tensor=None,
                 cond_on_img=False,
                 cond_img_dim = None
                 ):
        '''
            Initialize the class.
            Input:
                - encoder: encoder module
                - decoder: decoder module
        '''
        # Initialize the class
        super(DiffusionNet, self).__init__()
        
        # Set the encoder and decoder modules
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,
                x:torch.Tensor,
                t:torch.Tensor,
                y:Optional[torch.Tensor]=None,
                cond_img:Optional[torch.Tensor]=None,
                lsm_cond:Optional[torch.Tensor]=None,
                topo_cond:Optional[torch.Tensor]=None
                ):
        '''
            Forward function for the class.
            Input:
                - x: input tensor
                - t: time embedding tensor 
                - y: label tensor
        '''
        # Encode the input x
        enc_fmaps = self.encoder(x, t=t, y=y, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond)
        # Decode the encoded input, using the encoded feature maps
        segmentation_mask = self.decoder(*enc_fmaps, t=t)
        return segmentation_mask






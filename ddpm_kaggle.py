import os, tqdm, random, torch
import numpy as np
import torch.nn as nn
#from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Union, Iterable, Tuple
from modules_kaggle import *
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    # Set DANRA variable for use
    var = 'temp'#'prcp'#
    # Set size of DANRA images
    n_danra_size = 128#128#
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)

    # Set paths to data
    data_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str

    SAVE_FIGS = False
    PATH_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures'

    epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_channels = 1
    first_fmap_channels = 64#n_danra_size #
    last_fmap_channels = 512 #2048
    output_channels = 1
    time_embedding = 256
    learning_rate = 1e-4 #1e-2
    min_lr = 1e-6
    weight_decay = 0.0
    n_timesteps = 550
    beta_min = 1e-4
    beta_max = 2e-2
    beta_scheduler = 'linear'
    batch_size = 20
    n_samples = 365
    cache_size = 365
    image_size = (n_danra_size, n_danra_size)
    #seed = random.randint(0, 2**32 - 1)

    from data_kaggle import DANRA_Dataset

    dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size)#, seed=seed)

    sample_img = dataset[0]
    print(type(sample_img))
    print('\n')
    print(f'shape: {sample_img.shape}')
    print(f'min pixel value: {sample_img.min()}')
    print(f'mean pixel value: {sample_img.mean()}')
    print(f'max pixel value: {sample_img.max()}')


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title('Sample Image, ' + var)
    img = sample_img.squeeze()#
    image = ax.imshow(img, cmap='viridis')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    
    plt.show()


class DiffusionUtils:
    def __init__(self, n_timesteps:int, beta_min:float, beta_max:float, device:str='cpu', scheduler:str='linear'):
        assert scheduler in ['linear', 'cosine'], 'scheduler must be linear or cosine'
        self.n_timesteps = n_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device
        self.scheduler = scheduler

        self.betas = self.betaSamples()
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def betaSamples(self):
        if self.scheduler == 'linear':
            return torch.linspace(start=self.beta_min, end=self.beta_max, steps=self.n_timesteps).to(self.device)
        
        elif self.scheduler == 'cosine':
            betas = []
            for i in reversed(range(self.n_timesteps)):
                T = self.n_timesteps - 1
                beta = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (1 + np.cos((i / T) * np.pi))
                betas.append(beta)

            return torch.Tensor(betas).to(self.device)
        
    def sampleTimesteps(self, size:int):
        return torch.randint(low=1, high=self.n_timesteps, size=(size,)).to(self.device)
    
    def noiseImage(self, x:torch.Tensor, t:torch.LongTensor):
        assert len(x.shape) == 4, 'x must be a 4D tensor'

        alpha_hat_sqrts = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        one_minus_alpha_hat_sqrt = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x).to(self.device)
        
        return (alpha_hat_sqrts * x) + (one_minus_alpha_hat_sqrt * noise), noise
    
    def sample(self, x:torch.Tensor, model:nn.Module):
        #x shape: (batch_size, channels, height, width)
        assert len(x.shape) == 4, 'x must be a 4D tensor'
        
        model.eval()

        with torch.no_grad():
            iterations = range(1, self.n_timesteps)

            for i in tqdm.tqdm(reversed(iterations)):
                t = (torch.ones(x.shape[0]) * i).long().to(self.device)

                alpha = self.alphas[t][:, None, None, None]
                beta = self.betas[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                one_minus_alpha = 1 - alpha
                one_minus_alpha_hat = 1 - alpha_hat

                predicted_noise = model(x, t)

                if i > 1:
                    noise = torch.randn_like(x).to(self.device)
                else:
                    noise = torch.zeros_like(x).to(self.device)


                x = (1 / torch.sqrt(alpha)) * (x - ((one_minus_alpha) / torch.sqrt(one_minus_alpha_hat)) * predicted_noise)
                x = x + (torch.sqrt(beta) * noise)

            return x
        

if __name__ == '__main__':
    T = n_timesteps
    n_steps = 50
    alpha_values = {}

    # print('\n')
    # for scheduler in ['linear', 'cosine']:
    #     print(f'Running {scheduler} beta scheduler...')
        
    #     diffusion = DiffusionUtils(T, beta_min, beta_max, scheduler=scheduler)
    #     alpha_values[scheduler] = diffusion.alphas

    #     fig, axs = plt.subplots(2, (T//(n_steps*2))+1, figsize=(15,8))
        
    #     axs.flatten()[0].imshow(sample_img.squeeze())
    #     axs.flatten()[0].set_title('Original Image')

    #     for idx, t in enumerate(range(n_steps-1, T, n_steps)):
    #         t = torch.Tensor([t]).long()
    #         x, _ = diffusion.noiseImage(sample_img.unsqueeze(0), t)
    #         axs.flatten()[idx+1].imshow(x.squeeze())
    #         axs.flatten()[idx+1].set_title(f'Image at t={t.item()}')

    #     fig.set_tight_layout(True)
    #     if SAVE_FIGS:
    #         fig.savefig(PATH_SAVE + f'/ddpm_kaggle_ex__{scheduler}_diffusion.png')

    #     print('\n')



    # fig, axs = plt.subplots(1, 2, figsize=(20, 4))

    # axs[0].plot(alpha_values['linear'])
    # axs[0].set_xlabel('timestep (t)')
    # axs[0].set_ylabel('alpha (1-beta)')
    # axs[0].set_title('alpha values of linear scheduling')

    # axs[1].plot(alpha_values['cosine'])
    # axs[1].set_xlabel('timestep (t)')
    # axs[1].set_ylabel('alpha (1-beta)')
    # axs[1].set_title('alpha values of cosine scheduling')

    # plt.show()


class TrainingPipeline:
    def __init__(self, model, lossfunc, optimizer, diffusion_utils:DiffusionUtils,
                 device='cpu', weight_init=True, custom_weight_initializer=None):
        self.device = device
        self.model = model.to(self.device)
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.diffusion_utils = diffusion_utils
        self.custom_weight_initializer = custom_weight_initializer

        if self.weight_init:
            if self.custom_weight_initializer:
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)

    def xavier_init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    def save_model(self, dirname='./model_params', filename='DDPM.pth.tar'):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        
        state_dicts = {
            'network_params':self.model.state_dict(),
            'optimizer_params':self.optimizer.state_dict()
        }

        return torch.save(state_dicts, os.path.join(dirname, filename))
    

    def train(self, dataloader, verbose=True):
        self.model.train()
        loss = 0
        for idx, images in tqdm.tqdm(enumerate(dataloader)):
            self.model.zero_grad()
            images = images.to(self.device)

            t = self.diffusion_utils.sampleTimesteps(images.shape[0])

            x_t, noise = self.diffusion_utils.noiseImage(images, t)

            predicted_noise = self.model(x_t, t)

            batch_loss = self.lossfunc(predicted_noise, noise)

            batch_loss.backward()
            self.optimizer.step()
            loss += batch_loss.item()

        loss = loss / (idx + 1)
        if verbose:
            print(f'Training Loss: {loss}')

        return loss
    

if __name__ == '__main__':
    print('\n')
    train_dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size)#, seed=seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    encoder = Encoder(input_channels, time_embedding, block_layers=[2, 2, 2, 2])
    decoder = Decoder(last_fmap_channels, output_channels, time_embedding, first_fmap_channels)
    model = DiffusionNet(encoder, decoder)

    diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, device, beta_scheduler)

    lossfunc = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    pipeline = TrainingPipeline(model, lossfunc, optimizer, diffusion_utils, device, weight_init=True)

    checkpoint_path = '../ModelCheckpoints/kaggle/input/ddpm-model-params/DDPM.pth.tar'
    if os.path.isfile(checkpoint_path):
        print('Loading pretrained weights...')
        checkpoint_state = torch.load(checkpoint_path, map_location=device)['network_params']
        pipeline.model.load_state_dict(checkpoint_state)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pipeline.optimizer, T_max=epochs, eta_min=min_lr, verbose=True)
    
    test_input = torch.randn(2,input_channels,*image_size).to(device)
    t_test = torch.Tensor([2, 5]).to(device)
    test = pipeline.model(test_input, t_test)

    print('\n')
    print(test.shape)
    print('\n')
    print(sum([i.numel() for i in pipeline.model.parameters()]))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f'Model is training on {torch.cuda.get_device_name()}\n\n')
        print(f'Model is using {torch.cuda.memory_allocated()} bytes of memory\n\n')

    train_losses = []

    best_loss = np.inf

    for epoch in range(epochs):
        print(f'\n\nEpoch {epoch+1} of {epochs}:\n\n')
        train_loss = pipeline.train(train_dataloader, verbose=True)
        train_losses.append(train_loss)
        print(f'\n\nTraining Loss: {train_loss}\n\n')

        if train_loss < best_loss:
            best_loss = train_loss
            pipeline.save_model()
            print(f'Model saved at epoch {epoch+1} with loss {best_loss}')

        if ((epoch + 1) % 10) == 0 or epoch == (epochs - 1):
            print('Generating samples...')
            n = 4
            x = torch.randn(n, input_channels, *image_size).to(device)
            generated_images = diffusion_utils.sample(x, pipeline.model)
            generated_images = generated_images.detach().cpu()

            fig, axs = plt.subplots(1, n, figsize=(8,3))

            for i in range(n):
                img = generated_images[i].squeeze()
                img = img.type(torch.uint8)
                axs[i].imshow(img)
            plt.show()

    best_model_path = os.path.join('./model_params', 'DDPM.pth.tar')
    best_model_state = torch.load(best_model_path, map_location=device)['network_params']

    pipeline.model.load_state_dict(best_model_state)

    print('Generating samples...')
    n = 4

    x = torch.randn(n, input_channels, *image_size).to(device)
    generated_images = diffusion_utils.sample(x, pipeline.model)
    generated_images = generated_images.cpu()

    fig, axs = plt.subplots(1, n, figsize=(8,3))

    for i in range(n):
        img = generated_images[i].squeeze()#
        axs[i].imshow(img)
        axs[i].set_title(f'Generated Image {i+1}')




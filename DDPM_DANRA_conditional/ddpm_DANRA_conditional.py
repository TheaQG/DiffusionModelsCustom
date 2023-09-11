import os, tqdm, random, torch
import numpy as np
import torch.nn as nn
#from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Union, Iterable, Tuple


# Import objects from other files in this repository
from data_DANRA_conditional import DANRA_Dataset
from modules_DANRA_conditional import *
from diffusion_DANRA_conditional import DiffusionUtils
from training_DANRA_conditional import TrainingPipeline

if __name__ == '__main__':
    print(torch.__version__)

    from multiprocessing import freeze_support
    freeze_support()

    # General settings for use

    # Set path to save figures
    SAVE_FIGS = False
    PATH_SAVE = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures'

    # Test model input<->output before running?
    TEST_MODEL = True

    PLOT_EPOCH_SAMPLES = True

    # Define DANRA data information 
    # Set variable for use
    var = 'temp'#'prcp'# 
    # Set size of DANRA images
    n_danra_size = 128# 256#
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)
    # Set paths to data
    data_dir_danra = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str

    n_files = 0
    for root, _, files in os.walk(data_dir_danra):
        for name in files:
            if name.endswith('.npz'):
                n_files += 1




    # Define data hyperparameters
    input_channels = 1
    output_channels = 1
    n_samples = n_files
    cache_size = n_files
    image_size = (64,64)#(32,32)#(n_danra_size, n_danra_size)#
    n_seasons = 4

    # Define model hyperparameters
    epochs = 60
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    first_fmap_channels = 64#n_danra_size #
    last_fmap_channels = 512 #2048
    time_embedding = 256
    learning_rate = 1e-2#3e-4 #1e-2
    min_lr = 1e-6
    weight_decay = 0.0

    # Define diffusion hyperparameters
    n_timesteps = 1000
    beta_min = 1e-4
    beta_max = 0.02
    beta_scheduler = 'linear'


    # Define the dataset from data_DANRA_downscaling.py
    train_dataset = DANRA_Dataset(data_dir_danra, image_size, n_samples, cache_size, scale=True)

    # Define the torch dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Define the seed for reproducibility, and set seed for torch, numpy and random
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set torch to deterministic mode, meaning that the same input will always produce the same output
    torch.backends.cudnn.deterministic = True
    # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
    torch.backends.cudnn.benchmark = False

    # Define the encoder and decoder from modules_DANRA_downscaling.py
    encoder = Encoder(input_channels, time_embedding, block_layers=[2, 2, 2, 2], num_classes=4)
    decoder = Decoder(last_fmap_channels, output_channels, time_embedding, first_fmap_channels)
    # Define the model from modules_DANRA_downscaling.py
    model = DiffusionNet(encoder, decoder)
    # Define the diffusion utils from diffusion_DANRA_downscaling.py
    diffusion_utils = DiffusionUtils(n_timesteps, beta_min, beta_max, device, beta_scheduler)
    # Define the loss function
    lossfunc = nn.MSELoss()
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Define the training pipeline from training_DANRA_downscaling.py
    pipeline = TrainingPipeline(model, lossfunc, optimizer, diffusion_utils, device, weight_init=True)
    # Define the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pipeline.optimizer, T_max=epochs, eta_min=min_lr, verbose=True)

    # Define the path to the pretrained model 
    checkpoint_dir = '../../ModelCheckpoints/DDPM_DANRA/'
    checkpoint_name = 'DDPM_conditional.pth.tar'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    # Check if the path exists
    if os.path.isfile(checkpoint_path):
        # If the path exists, load the pretrained weights
        print('Loading pretrained weights...')
        checkpoint_state = torch.load(checkpoint_path, map_location=device)['network_params']
        pipeline.model.load_state_dict(checkpoint_state)

    if TEST_MODEL:
        print('\nTesting model output size...\n')
        test_input = torch.randn(4,input_channels,*image_size).to(device)
        t_test = torch.Tensor([1, 5, 7, 11]).to(device)
        y_test = torch.zeros(n_seasons)
        idx_test = np.random.randint(0, n_seasons)
        y_test[idx_test] = 1
        y_test = y_test.type(torch.int64).to(device)

        test = pipeline.model(test_input, t_test, y_test)

        print('\n')
        print(f'Output shape of test: {test.shape}')
        print('\n')
        print('Number of parameters: ' + str(sum([i.numel() for i in pipeline.model.parameters()])))

    # Check if device is cuda, if so print information and empty cache
    if torch.cuda.is_available():
        print(f'Model is training on {torch.cuda.get_device_name()}\n\n')
        print(f'Model is using {torch.cuda.memory_allocated()} bytes of memory\n\n')
        torch.cuda.empty_cache()

    # Set empty list for train losses
    train_losses = []

    # Set best loss to infinity
    best_loss = np.inf

    # Loop over epochs
    for epoch in range(epochs):
        
        print(f'\n\nEpoch {epoch+1} of {epochs}:\n\n')
        if epoch == 0:
            PLOT_FIRST_IMG = True
        else:
            PLOT_FIRST_IMG = False

        # Call the train function from pipeline
        train_loss = pipeline.train(train_dataloader, verbose=True, PLOT_FIRST=PLOT_FIRST_IMG, SAVE_PATH=PATH_SAVE)
        # Append train loss to list
        train_losses.append(train_loss)
        
        print(f'\n\nTraining Loss: {train_loss}\n\n')

        # If train loss is better than best loss, save model
        if train_loss < best_loss:
            best_loss = train_loss
            pipeline.save_model(checkpoint_dir, checkpoint_name)
            print(f'Model saved at epoch {epoch+1} with loss {best_loss}')

        # If epoch is multiple of 10 or last epoch, generate samples
        if (((epoch + 1) % 10) == 0 and PLOT_EPOCH_SAMPLES) or (epoch == (epochs - 1) and PLOT_EPOCH_SAMPLES):
            print('Generating samples...')
            n = 4
            # Generate random fields of batchsize n
            x = torch.randn(n, input_channels, *image_size).to(device)

            # Generate random season labels of batchsize n
            y = torch.randint(0, 4, (n,)).to(device) # 4 seasons, 0-3

            # Sample generated images from model
            generated_images = diffusion_utils.sample(x, pipeline.model, y)
            generated_images = generated_images.detach().cpu()

            # Plot generated images
            fig, axs = plt.subplots(1, n, figsize=(8,3))

            for i in range(n):
                img = generated_images[i].squeeze()
                axs[i].imshow(img, vmin=-1, vmax=1, cmap='viridis')
                axs[i].set_title(f'Season: {y[i].item()}')
            fig.tight_layout()
            fig.savefig(PATH_SAVE + '/Samples/Samples_64x64/Generated_samples_64x64__v2__epoch_' + str(epoch+1) + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

    # Load best model state
    best_model_path = checkpoint_path#os.path.join('../../ModelCheckpoints/DDPM_DANRA', 'DDPM.pth.tar')
    best_model_state = torch.load(best_model_path)['network_params']

    # Load best model state into model
    pipeline.model.load_state_dict(best_model_state)

    print('Generating samples...')

    # Set number of samples to generate
    n = 4

    # Generate samples
    x = torch.randn(n, input_channels, *image_size).to(device)

    # Generate random season labels of batchsize n
    y = torch.randint(0, 4, (n,)).to(device) # 4 seasons, 0-3

    # Sample generated images from model
    generated_images = diffusion_utils.sample(x, pipeline.model, y)
    generated_images = generated_images.detach().cpu()

    # Plot samples
    fig, axs = plt.subplots(1, n, figsize=(8,3))

    for i in range(n):
        img = generated_images[i].squeeze()
        axs[i].imshow(img, vmin=-1, vmax=1, cmap='viridis')
        axs[i].set_title(f'Generated Image {i+1}, season: {y[i].item()}')

    fig.savefig(PATH_SAVE + f'/Samples/Samples_64x64/ddpm_kaggle_ex__{var}_generated_samples_64x64__v2__conditional.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    import pickle

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(train_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    fig.savefig(PATH_SAVE + f'/ddpm_kaggle_ex__{var}_training_loss_conditional_64x64.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

    with open('train_losses_conditional', 'wb') as fp:
        pickle.dump(train_losses, fp)





# if __name__ == '__main__':

    

#     sample_img = dataset[0]
#     print(type(sample_img))
#     print('\n')
#     print(f'shape: {sample_img.shape}')
#     print(f'min pixel value: {sample_img.min()}')
#     print(f'mean pixel value: {sample_img.mean()}')
#     print(f'max pixel value: {sample_img.max()}')


#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     ax.set_title('Sample Image, ' + var)
#     img = sample_img.squeeze()#
#     image = ax.imshow(img, cmap='viridis')
#     fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    
#     plt.show()

#     T = n_timesteps
#     n_steps = 50
#     alpha_values = {}

#     # print('\n')
#     # for scheduler in ['linear', 'cosine']:
#     #     print(f'Running {scheduler} beta scheduler...')
        
#     #     diffusion = DiffusionUtils(T, beta_min, beta_max, scheduler=scheduler)
#     #     alpha_values[scheduler] = diffusion.alphas

#     #     fig, axs = plt.subplots(2, (T//(n_steps*2))+1, figsize=(15,8))
        
#     #     axs.flatten()[0].imshow(sample_img.squeeze())
#     #     axs.flatten()[0].set_title('Original Image')

#     #     for idx, t in enumerate(range(n_steps-1, T, n_steps)):
#     #         t = torch.Tensor([t]).long()
#     #         x, _ = diffusion.noiseImage(sample_img.unsqueeze(0), t)
#     #         axs.flatten()[idx+1].imshow(x.squeeze())
#     #         axs.flatten()[idx+1].set_title(f'Image at t={t.item()}')

#     #     fig.set_tight_layout(True)
#     #     if SAVE_FIGS:
#     #         fig.savefig(PATH_SAVE + f'/ddpm_kaggle_ex__{scheduler}_diffusion.png')

#     #     print('\n')



#     # fig, axs = plt.subplots(1, 2, figsize=(20, 4))

#     # axs[0].plot(alpha_values['linear'])
#     # axs[0].set_xlabel('timestep (t)')
#     # axs[0].set_ylabel('alpha (1-beta)')
#     # axs[0].set_title('alpha values of linear scheduling')

#     # axs[1].plot(alpha_values['cosine'])
#     # axs[1].set_xlabel('timestep (t)')
#     # axs[1].set_ylabel('alpha (1-beta)')
#     # axs[1].set_title('alpha values of cosine scheduling')

#     # plt.show()



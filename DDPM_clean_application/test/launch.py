import yaml
import argparse
from ..src.main_ddpm import main_ddpm
from ..src.utils import convert_npz_to_zarr

def data_checker(args):
    '''
        Based on arguments check if data exists and in the right format
        If not, create right format
    '''
    hr_var = args.HR_VAR
    lr_vars = args.LR_VARS

    data_path = args.path_data

    # Check if the data exists


# --path_data '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/'
# --path_save '/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_clean/'

def launch_from_args():
    '''
        Launch the training from the command line arguments
    '''
    

    parser = argparse.ArgumentParser(description='Train a model for the downscaling of climate data')
    parser.add_argument('--HR_VAR', type=str, default='temp', help='The high resolution variable')
    parser.add_argument('--HR_SHAPE', type=list, default=(32, 32), help='The shape of the high resolution data')
    parser.add_argument('--LR_VARS', type=list, default=['temp'], help='The low resolution variables')
    parser.add_argument('--LR_SHAPE', type=list, default=(32, 32), help='The shape of the low resolution data')
    parser.add_argument('--scaling', type=bool, default=True, help='Whether to scale the data')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--path_save', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_clean/', help='The path to save the results')
    parser.add_argument('--path_checkpoint', type=str, default='model_checkpoints/', help='The path to the checkpoints')
    parser.add_argument('--in_channels', type=int, default=1, help='The number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='The number of output channels')
    parser.add_argument('--season_shape', type=list, default=(4,), help='The shape of the season data')
    parser.add_argument('--loss_type', type=str, default='sdfweighted', help='The type of loss function')
    parser.add_argument('--config_name', type=str, default='ddpm', help='The name of the configuration file')
    parser.add_argument('--create_figs', type=bool, default=True, help='Whether to create figures')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('--first_fmap_channels', type=int, default=32, help='The number of channels in the first feature map')
    parser.add_argument('--last_fmap_channels', type=int, default=512, help='The number of channels in the last feature map')
    parser.add_argument('--time_embedding_size', type=int, default=256, help='The size of the time embedding')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='The minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='The weight decay')
    parser.add_argument('--n_timesteps', type=int, default=4, help='The number of timesteps in diffusion process')
    parser.add_argument('--beta_range', type=list, default=[0.00001, 0.02], help='The range of beta values')
    parser.add_argument('--beta_scheduler', type=str, default='cosine', help='The beta scheduler')
    parser.add_argument('--noise_variance', type=float, default=0.01, help='The noise variance')
    parser.add_argument('--CUTOUTS', type=bool, default=True, help='Whether to use cutouts')
    parser.add_argument('--CUTOUT_DOMAINS', type=list, default=[(170, 350, 340, 520)], help='The cutout domains')
    parser.add_argument('--num_workers', type=int, default=1, help='The number of workers')
    parser.add_argument('--cache_size', type=int, default=0, help='The cache size')
    parser.add_argument('--data_split_type', type=str, default='random', help='The type of data split')
    parser.add_argument('--data_split_params', type=dict, default={'train_size': 0.8, 'val_size': 0.1, 'test_size': 0.1}, help='The data split parameters')
    parser.add_argument('--n_gen_samples', type=int, default=4, help='The number of generated samples')
    parser.add_argument('--num_heads', type=int, default=1, help='The number of heads')
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', help='The learning rate scheduler')
    parser.add_argument('--lr_scheduler_params', type=dict, default={'factor': 0.5, 'patience': 5, 'threshold': 0.01, 'min_lr': 1e-6}, help='The learning rate scheduler parameters')
    parser.add_argument('--early_stopping', type=bool, default=True, help='Whether to use early stopping')
    parser.add_argument('--early_stopping_params', type=dict, default={'patience': 10, 'min_delta': 0.001}, help='The early stopping parameters')
    parser.add_argument('--device', type=str, default='cpu', help='The device')
    #parser.add_argument('--init_weights', type=str, default='kaiming', help='The initialization weights')

    args = parser.parse_args()

    # Launch the training
    main_ddpm(args)







if __name__ == '__main__':
    launch_from_args()
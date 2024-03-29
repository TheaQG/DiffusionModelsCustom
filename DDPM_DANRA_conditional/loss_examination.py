import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from data_DANRA_conditional import preprocess_lsm_topography_from_data


if __name__ == '__main__':
    print('\n\n')

    LOSS_PATH = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses/Losses/'

    # Load the losses with pickle
    file = "/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses/Losses/" \
            + "Training_losses__temp__64x64__unconditional_random__simple__None_seasons_ValidSplitInTime_9yrs_train"
    L_train_unconditional = pd.read_pickle(file)
    file = "/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses/Losses/"\
            + "Training_losses__temp__64x64__unconditional_random__simple__None_seasons_ValidSplitInTime_9yrs_valid"
    L_val_unconditional = pd.read_pickle(file)

    file = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses/Losses/Training_losses__temp__64x64__cond_lsm_topo_only_random__sdfweighted__4_seasons_ValidSplitInTime_9yrs_train'    
    L_train_lsm_topo_only = pd.read_pickle(file)
    file = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses/Losses/Training_losses__temp__64x64__cond_lsm_topo_only_random__sdfweighted__4_seasons_ValidSplitInTime_9yrs_valid'
    L_val_lsm_topo_only = pd.read_pickle(file)

    file = "/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses/Losses/Training_losses__temp__64x64__uniform_cond_lsm_topo_random__sdfweighted__4_seasons_ValidSplitInTime_9yrs_train"
    L_train_uniform = pd.read_pickle(file)
    file = "/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses/Losses/Training_losses__temp__64x64__uniform_cond_lsm_topo_random__sdfweighted__4_seasons_ValidSplitInTime_9yrs_valid"
    L_val_uniform = pd.read_pickle(file)

    file = "/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses/Losses/Training_losses__temp__64x64__ERA5_cond_lsm_topo_random__sdfweighted__4_seasons_ValidSplitInTime_9yrs_ValLoss_train"
    L_train_era5 = pd.read_pickle(file)
    file = "/Users/au728490/Documents/PhD_AU/Python_Scripts/Losses/Losses/Training_losses__temp__64x64__ERA5_cond_lsm_topo_random__sdfweighted__4_seasons_ValidSplitInTime_9yrs_ValLoss_valid"
    L_val_era5 = pd.read_pickle(file)


    # Make a plot with all losses full, and zoom in on smaller area
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(L_train_unconditional, label='Unconditional\ntraining', color='black', linestyle='-')
    ax.plot(L_val_unconditional, label='Unconditional\nvalidation', color='black', linestyle='-',alpha=0.5, lw=1)
    ax.plot(L_train_lsm_topo_only, label='LSM + topo only\ntraining', color='blue', linestyle='-')
    ax.plot(L_val_lsm_topo_only, label='LSM + topo only\nvalidation', color='blue', linestyle='-',alpha=0.5, lw=1)
    ax.plot(L_train_uniform, label='Uniform\ntraining', color='red', linestyle='-')
    ax.plot(L_val_uniform, label='Uniform\nvalidation', color='red', linestyle='-',alpha=0.5, lw=1)
    ax.plot(L_train_era5, label='ERA5\ntraining', color='green', linestyle='-')
    ax.plot(L_val_era5, label='ERA5\nvalidation', color='green', linestyle='-',alpha=0.5, lw=1)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.set_ylim([0.02, .9])
    ax.set_xlim([-2,100])
    # Set tick font size for both axis
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title('Training and validation loss for different conditional models', fontsize=18)
    ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))

    # Add zoomed in area
    axins = ax.inset_axes([0.35, 0.35, 0.6, 0.6])
    axins.plot(L_train_unconditional, label='Unconditional training', color='black', linestyle='-')
    axins.plot(L_val_unconditional, label='Unconditional validation', color='black', linestyle='-', alpha=0.5, lw=1)
    axins.plot(L_train_lsm_topo_only, label='LSM and topo only training', color='blue', linestyle='-')
    axins.plot(L_val_lsm_topo_only, label='LSM and topo only validation', color='blue', linestyle='-', alpha=0.5, lw=1)
    axins.plot(L_train_uniform, label='Uniform training', color='red', linestyle='-')
    axins.plot(L_val_uniform, label='Uniform validation', color='red', linestyle='-', alpha=0.5, lw=1)
    axins.plot(L_train_era5, label='ERA5 training', color='green', linestyle='-')
    axins.plot(L_val_era5, label='ERA5 validation', color='green', linestyle='-', alpha=0.5, lw=1)
    axins.set_xlim(2, 50)
    axins.set_ylim(0.025, 0.2)
    # axins.set_xticklabels('')
    # axins.set_yticklabels('')
    ax.indicate_inset_zoom(axins)

    fig.tight_layout()

    fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/DDPM_evaluation/Losses.png', dpi=600, bbox_inches='tight')


    # Plot the losses zommed in and zoom further in
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(L_train_unconditional, label='Unconditional training', color='black', linestyle='-')
    ax.plot(L_val_unconditional, label='Unconditional validation', color='black', linestyle='-', alpha=0.5, lw=1)
    ax.plot(L_train_lsm_topo_only, label='LSM and topo only training', color='blue', linestyle='-')
    ax.plot(L_val_lsm_topo_only, label='LSM and topo only validation', color='blue', linestyle='-', alpha=0.5, lw=1)
    ax.plot(L_train_uniform, label='Uniform training', color='red', linestyle='-')
    ax.plot(L_val_uniform, label='Uniform validation', color='red', linestyle='-', alpha=0.5, lw=1)
    ax.plot(L_train_era5, label='ERA5 training', color='green', linestyle='-')
    ax.plot(L_val_era5, label='ERA5 validation', color='green', linestyle='-', alpha=0.5, lw=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_ylim([0.025, 0.2])
    ax.set_xlim([0,80])
    ax.set_title('Training and validation loss for different conditional models')
    ax.legend(loc='upper left', fontsize='small')

    # Add zoomed in area
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.plot(L_train_unconditional, label='Unconditional training', color='black', linestyle='-')
    axins.plot(L_val_unconditional, label='Unconditional validation', color='black', linestyle='-', alpha=0.5, lw=1)
    axins.plot(L_train_lsm_topo_only, label='LSM and topo only training', color='blue', linestyle='-')
    axins.plot(L_val_lsm_topo_only, label='LSM and topo only validation', color='blue', linestyle='-', alpha=0.5, lw=1)
    axins.plot(L_train_uniform, label='Uniform training', color='red', linestyle='-')
    axins.plot(L_val_uniform, label='Uniform validation', color='red', linestyle='-', alpha=0.5, lw=1)
    axins.plot(L_train_era5, label='ERA5 training', color='green', linestyle='-')
    axins.plot(L_val_era5, label='ERA5 validation', color='green', linestyle='-', alpha=0.5, lw=1)
    axins.set_xlim(5, 50)
    axins.set_ylim(0.05, 0.1)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax.indicate_inset_zoom(axins)

    fig.tight_layout()
    plt.show()

    fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/DDPM_evaluation/Losses_zoomed_in.png', dpi=600, bbox_inches='tight')
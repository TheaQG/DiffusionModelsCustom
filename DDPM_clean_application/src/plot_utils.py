'''
    This file contains utility helper functions for plotting.    

    Need to implement:
    - plot_sample (WIP): function for plotting samples from the dataset
    - plot_generated: function for plotting generated samples
    - plot_sample_and_generated: function for plotting samples and generated samples
    - plot_distributions: function for plotting distributions of test and generated samples
    - plot_diffusion: function for plotting diffusion process
        - forward_diffusion
        - backward_diffusion
    - plot_loss: function for plotting loss during training
    - plot_metrics: function for plotting metrics 
    

'''

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_sample(dataloader,
                variable:str,
                n_samples:int=1,
                device:str='cpu',
                save_figs:bool=False,
                show_figs:bool=False,
                SAVE_PATH:str=None,
                SAVE_NAME:str=None,
                return_figs:bool=False):
    '''
        Function for plotting samples from the dataset - independent of sample type
    '''

    if variable == 'temp':
        cmap_name = 'plasma'
        cmap_label = 'Temperature [°C]'
    elif variable == 'prcp':
        cmap_name = 'inferno'
        cmap_label = 'Precipitation [mm/day]'

    for idx, samples in enumerate(dataloader):
        # Go through possible keys and send to device if available
        if 'img' in samples.keys():
            images = samples['img'].to(device)
            images = samples['img'].to(torch.float)
            # Print type of image tensor
            #print(f'Images are of type: {images.dtype}')
        else:
            # Print error, you need to have images
            print('Error: you need to have images in your dataset')

        if 'classifier' in samples.keys() and samples['classifier'] is not None:
            seasons = samples['classifier'].to(device)
            # Print type of seasons
            #print(f'Seasons are of type: {seasons.dtype}')
        else:
            seasons = None

        if 'img_cond' in samples.keys() and samples['img_cond'] is not None:
            # Check the data type of the samples
            cond_images = samples['img_cond'].to(device)
            cond_images = samples['img_cond'].to(torch.float)
            # Print type of cond_images
            #print(f'Conditional images are of type: {cond_images.dtype}')
        else:
            cond_images = None

        if 'lsm' in samples.keys() and samples['lsm'] is not None:
            lsm = samples['lsm'].to(device)
            lsm = samples['lsm'].to(torch.float)
            # Print type of lsm
            #print(f'LSM is of type: {lsm.dtype}')
        else:
            lsm = None

        if 'sdf' in samples.keys() and samples['sdf'] is not None:
            sdf = samples['sdf'].to(device)
            sdf = samples['sdf'].to(torch.float)
            # Print type of sdf
            #print(f'SDF is of type: {sdf.dtype}')
        else:
            sdf = None

        if 'topo' in samples.keys() and samples['topo'] is not None:
            topo = samples['topo'].to(device)
            topo = samples['topo'].to(torch.float)
            # Print type of topo
            #print(f'Topo is of type: {topo.dtype}')
        else:
            topo = None


        if 'img' in samples.keys():
            images_samples = images[:n_samples]
        if 'classifier' in samples.keys():
            seasons_samples = seasons[:n_samples]
        if 'img_cond' in samples.keys():
            cond_images_samples = cond_images[:n_samples]
        if 'lsm' in samples.keys():
            lsm_samples = lsm[:n_samples]
        if 'sdf' in samples.keys():
            sdf_samples = sdf[:n_samples]
        if 'topo' in samples.keys():
            topo_samples = topo[:n_samples]

        # Create lists to store fig and ax objects
        figs_out = []
        axs_out = []

        if 'img' in samples.keys():
            fig, axs = plt.subplots(1, n_samples, figsize=(18, 4))
            fig.suptitle('DANRA images')
            i = 0
            for im, ax in zip(images_samples, axs.flatten()):
                image = ax.imshow(im.permute(1,2,0).cpu().detach().numpy(), cmap=cmap_name)

                if 'classifier' in samples.keys():
                    ax.set_title(f'Season: {seasons_samples[i].item()}')
                else:
                    ax.set_title(f'Image {idx}')
                #ax.axis('off')
                ax.set_ylim([0, im.shape[1]])
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=cmap_label, orientation='vertical')
                i += 1
            fig.tight_layout()
            figs_out.append(fig)
            axs_out.append(axs)
            

        if 'img_cond' in samples.keys():
            fig2, ax2s = plt.subplots(1, n_samples, figsize=(18, 4))
            fig2.suptitle('Conditional ERA5 images')
            i = 0
            for cond_im, ax in zip(cond_images_samples, ax2s.flatten()):
                image = ax.imshow(cond_im.permute(1,2,0).cpu().detach().numpy(), cmap=cmap_name)

                if 'classifier' in samples.keys():
                    ax.set_title(f'Season: {seasons_samples[i].item()}')
                else:
                    ax.set_title(f'Conditional image {idx}')
                #ax.axis('off')
                ax.set_ylim([0, cond_im.shape[1]])
                fig2.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=cmap_label, orientation='vertical')
                i += 1
            fig2.tight_layout()
            figs_out.append(fig2)
            axs_out.append(ax2s)
            



        if 'sdf' or 'lsm' or 'topo' in samples.keys():
            # Count how many of the keays are present and create new dict with only the present keys
            n_keys = 0
            geo_dict = {}
            if 'lsm' in samples.keys() and samples['lsm'] is not None:
                n_keys += 1
                geo_dict['lsm'] = lsm_samples
            if 'topo' in samples.keys() and samples['topo'] is not None:
                n_keys += 1
                geo_dict['topo'] = topo_samples
            if 'sdf' in samples.keys() and samples['sdf'] is not None:
                n_keys += 1
                geo_dict['sdf'] = sdf_samples

            if n_keys > 0:
                fig3, ax3s = plt.subplots(n_keys, n_samples, figsize=(18, 6))

                # Loop through keys and samples for each key
                i = 0
                for key, data in geo_dict.items():
                    for geo_sample, ax in zip(data, ax3s[i,:]):
                        image = ax.imshow(geo_sample.permute(1,2,0).cpu().detach().numpy())
                        
                        ax.axis('off')
                        ax.set_ylim([0, geo_sample.shape[1]])
                    i += 1
                fig3.tight_layout()

                if save_figs:
                    fig3.savefig(SAVE_PATH + '/' + 'topo_lsm_' + SAVE_NAME, dpi=600, bbox_inches='tight')
                
                figs_out.append(fig3)
                axs_out.append(ax3s)
        
        if save_figs:
            print(f'\n\n\nSaving {n_samples} upsampled training images...\n\n\n')
    
            if 'img' in samples.keys() and samples['img'] is not None:
                fig.savefig(SAVE_PATH + '/' + SAVE_NAME, dpi=600, bbox_inches='tight')
            if 'img_cond' in samples.keys() and samples['img_cond'] is not None:
                fig2.savefig(SAVE_PATH + '/' + 'cond_ERA5_' + SAVE_NAME, dpi=600, bbox_inches='tight')
                

        if show_figs:
            # Show figure for ten seconds
            plt.show()
            plt.pause(10)
            plt.close('all')
    
    if return_figs:
        return figs_out, axs_out
    
def plot_generated():
    pass
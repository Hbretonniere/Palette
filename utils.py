import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.neighbors import KernelDensity

def show_rgb(x):
    plt.imshow([[(x[0], x[1], x[2])]])

def transfo_rgb(x):
    return [[(x[0], x[1], x[2])]]


def density_3D(points, bandwidth):

    # Create the Kernel Density Estimation object
    kde = KernelDensity(bandwidth=bandwidth, metric='euclidean', kernel='gaussian', algorithm='ball_tree')

    # Fit the KDE to the points
    kde.fit(points)

    # Evaluate the KDE at each point to get the estimated density
    densities = kde.score_samples(points)
    densities = np.exp(densities)  # convert log density to actual density
    return densities


def compute_and_show(painting_name, nb_color=10, similarity=100,
                     degrade=10, clip=0, band_width=0.5):
    try:
        image = Image.open(f'paintings/{painting_name}.jpg')
    except:
        image = Image.open(f'paintings/{painting_name}.png')

    image = np.asarray(image)[:, :, :3]
    low_res = downscale_local_mean(image, (degrade, degrade, 1))
    clipped = clip_black(low_res, clip)
    flat_img = clipped.reshape(clipped.shape[0]*clipped.shape[1], 3)

    mid_colors, counts, density_colors = find_colors(flat_img, similarity,
                                                     band_width)
    painting = plt.figure()
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    palette = show_palette(mid_colors, counts, density_colors, nb_color)
    return painting, palette 


def show_palette(mid_colors, counts, density_colors, nb_colors=10):
    fig, ax = plt.subplots(1, nb_colors)#, figsize=(15, 5))
    # fig, ax = plt.subplots(2, nb_colors)
    sorted_count = np.sort(counts)
    n_colors_found = len(mid_colors)
    dim_cross = 24
    cross = np.zeros((dim_cross, dim_cross))
    # Change the values along the main diagonal
    diag_indices = np.diag_indices(dim_cross)
    cross[diag_indices] = 1
    secondary_diag_indices = np.diag_indices(dim_cross)[0], np.diag_indices(dim_cross)[1][::-1]
    cross[secondary_diag_indices] = 1
    
    for i in range(1, nb_colors+1):
        if i > n_colors_found-3:

            # ax[0, i-1].imshow(cross, cmap='Greys')
            ax[i-1].imshow(cross, cmap='Greys')

        else:

            # ax[0, i-1].imshow(transfo_rgb(mid_colors[np.where(counts==sorted_count[-i])[0][0]]/255))
            ax[i-1].imshow(transfo_rgb(mid_colors[np.where(counts==sorted_count[-i])[0][0]]/255))
            # ax[1, i-1].imshow(transfo_rgb(density_colors[np.where(counts==sorted_count[-i])[0][0]]/255))
        # ax[0, i-1].set_xticks([])
        # ax[0, i-1].set_yticks([])
        # ax[1, i-1].set_xticks([])
        # ax[1, i-1].set_yticks([])
        ax[i-1].set_xticks([])
        ax[i-1].set_yticks([])


    return fig

    

def clip_black(image, value=0.1, show=False):
    """ Clip black values uch that black or very dark colors
    are not over-reprsented in the image
     
    params: 
        - image: np.array ; the rgb image 
        - value: float ; the rgb value to create the clipped black
    """
    # print(image[0, 0])
    clipped = np.copy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # try:
                if all(image[i, j]/255 < value):
    #                 print('clipped')
                    clipped[i, j] = [value * 255,
                                value * 255,
                                value * 255]
            # except TypeError:
            #     continue
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(10, 20))
        ax[0].imshow(image)
        ax[1].imshow(clipped)
    return clipped

def lower_resolution(image, factor):
     image = image[::factor, ::factor]
     return image


def find_colors(flat_img, similarity, band_width, show=False):
    # image = clip_black(image, 0.2)
    # low_res = lower_resolution(image, 10)
    # flat_img = low_res.reshape(low_res.shape[0]*low_res.shape[1], 3)
    counts = np.zeros((np.shape(flat_img)[0]))
    mid_colors = []
    density_colors = []
    pix_i = 0

    # loop until we have remove all the image pixels
    while(len(flat_img) > 0):
        pix_i += 1
        list_sim = []
        sum_color = [0, 0, 0]
        sim_colors = []
        try:
            for j in range(1, len(flat_img)):
                ''' check if the distance between the current pixel (pix_i)
                and the compared pixel is lower than
                the accepted similarity'''

                # if np.sum(flat_img[j]-flat_img[pix_i]) < similarity:
                # print(flat_img[j].shape)
                # print(flat_img[pix_i].shape) 
                if np.linalg.norm(flat_img[j]-flat_img[pix_i]) < similarity: 
                    ''' if the pixel is the similar, we add it to the list
                    of similar pixels '''
                    list_sim.append(j)
                    sum_color += flat_img[j]
                    sim_colors.append(flat_img[j])
                    counts[pix_i] += 1
       
            ''' after finding all pixels similar to pix_i, compute
            the median color ''' 
            # mid_colors.append(np.array(sum_color) / len(list_sim))
            # print()
            ''' after finding all pixels similar to pix_i, compute
            the median color ''' 
            # mid_colors.append(np.array(sum_color) / len(list_sim))
            # print('mean', mid_colors)
            # print('mean', mid_colors.shape)
            # print('shape', np.shape(sim_colors))
            # print(np.median(sim_colors, axis=0))
            mid_colors.append(np.median(sim_colors, axis=0))
            # band_width = 0.5
            # band_width = 
            # densities = density_3D(sim_colors, band_width)

            # density_colors.append(sim_colors[np.argmax(densities)])
            

            # if show:
            #     fig, ax = plt.subplots(1, 8, figsize=(20, 5))
            #     ax[0].imshow(transfo_rgb(flat_img[i]))
            #     ax[0].set_title(flat_img[i])
            #     ax[-1].imshow(transfo_rgb(mid_color[-1]/255))
            #     ax[-1].set_title(len(list_sim))
            #     for k in range(6):
            #         try:
            #             ax[k+1].imshow(transfo_rgb(flat_img[list_sim[k]]))
            #             ax[k+1].set_title(flat_img[list_sim[k]])
            #         except:
            #             ax[k+1].imshow(transfo_rgb([255, 255, 255]))

            ''' Delete all the found pixels, so that we can go to the
            next color still present in the painting '''
            flat_img = np.delete(flat_img, list_sim, 0)
            flat_img = np.delete(flat_img, 0, 0)
            
    #         print(len(flat_img))
        except IndexError:
            return mid_colors, counts, density_colors            
    return mid_colors, counts, density_colors
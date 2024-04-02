import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import downscale_local_mean
from sklearn.neighbors import KernelDensity
import matplotlib.animation as animation


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


def animation_traject(i, image, color_groups):
    
    return image


# def compute_positions(image, colors, n_colors):
#     pix_shape = 4
#     fraction = 1
#     start_y = [0, 200, 400, 600, 800, 1000, 1200]
#     for i in range(n_colors):
#         # print(np.shape(colors[i])[0])\
#         # print(np.shape(colors))
#         colors_to_plot = int(fraction*np.shape(colors[i])[0])
#         random_indices = np.random.randint(0, np.shape(colors[i])[0],
#                                            colors_to_plot)
#         # print(random_indices)
#         # selected_colors = colors[i][random_indices]
#         k = 0
#         for x in range(0,
#                        pix_shape*int(np.sqrt(colors_to_plot)))[::pix_shape]:
#             for y in range(0,
#                            pix_shape*int(np.sqrt(colors_to_plot)))[::pix_shape]:
                
#                 image[x:x+pix_shape, y+start_y[i]:y+start_y[i]+pix_shape] = colors[i][random_indices[k]]
#                 k += 1

def display_color_groups(canvas, colors, n_colors):
    pix_shape = 4
    fraction = 1
    start_y = [0, 200, 400, 600, 800, 1000, 1200]
    for i in range(n_colors):
        # print(np.shape(colors[i])[0])\
        # print('shaoe', colors.shape)
        colors_to_plot = int(fraction*np.shape(colors[i])[0])
        random_indices = np.random.randint(0, np.shape(colors[i])[0],
                                           colors_to_plot)
        # print(random_indices)
        # selected_colors = colors[i][random_indices]
        k = 0
        for x in range(0,
                       pix_shape*int(np.sqrt(colors_to_plot)))[::pix_shape]:
            for y in range(0,
                           pix_shape*int(np.sqrt(colors_to_plot)))[::pix_shape]:
                canvas[x:x+pix_shape, y+start_y[i]:y+start_y[i]+pix_shape] = colors[i][random_indices[k]]
                k += 1



def display_painting(image, colors, n_colors, animate, margin=0.3):
    painting_canevas = plt.figure()
    # print(np.shape(colors))
    # n_colors = 
    if animate:
    
        sx, sy = np.shape(image)[0], np.shape(image)[1]
        
        margin_sx, margin_sy = int(margin*sx), int(margin*sy) 
        # new_image = np.zeros((sx+2*margin_sx, sy+2*margin_sy, 3)) + 10
        new_image = np.random.randint(0, 1, (sx+2*margin_sx, sy+2*margin_sy, 3))
        new_image[margin_sx:margin_sx+sx, margin_sy:margin_sy+sy] = image

        display_color_groups(new_image, colors, n_colors)
        plt.imshow(new_image)
    
    else:
        plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    return painting_canevas


def compute_and_show(painting_name, nb_color=10, similarity=50,
                     degrade=10, clip=0, band_width=0.5, animate=False):
    try:
        image = Image.open(f'paintings/{painting_name}.jpg')
    except:
        image = Image.open(f'paintings/{painting_name}.png')

    image = np.asarray(image)[:, :, :3]
    low_res = downscale_local_mean(image, (degrade, degrade, 1))
    imshape = image.shape[::2]
    clipped = clip_black(low_res, clip)
    flat_img = clipped.reshape(clipped.shape[0]*clipped.shape[1], 3)

    median_colors, color_groups, counts, density_colors = find_colors(flat_img, similarity,
                                                     band_width,
                                                     imshape)
    # color_group_postions = compute_positions(image, color_groups, nb_color)
    painting = display_painting(image, color_groups, nb_color, animate)
    palette = show_palette(color_groups, counts, density_colors, nb_color)
    
    return painting, palette 


def show_palette(color_groups, counts, density_colors, nb_colors=10):
    fig, ax = plt.subplots(1, nb_colors)#, figsize=(15, 5))
    # fig, ax = plt.subplots(2, nb_colors)
    # print(counts)
    # sorted_count = np.sort(counts)
    n_colors_found = len(counts)
    dim_cross = 24
    cross = np.zeros((dim_cross, dim_cross))
    # Change the values along the main diagonal
    diag_indices = np.diag_indices(dim_cross)
    cross[diag_indices] = 1
    secondary_diag_indices = np.diag_indices(dim_cross)[0], np.diag_indices(dim_cross)[1][::-1]
    cross[secondary_diag_indices] = 1
    # print(np.shape(median_colors), n_colors_found, np.shape(counts))
    for i in range(0, nb_colors):
        # print(n_colors_found, i)
        if i < n_colors_found:
            color = transfo_rgb(np.median(color_groups[i], axis=0)/255)
            ax[i].imshow(color)
            # ax[i-1].imshow(transfo_rgb(median_colors[i]/255))
        # if i > n_colors_found-2:

        else:
            # ax[0, i-1].imshow(cross, cmap='Greys')
            ax[i-1].imshow(cross, cmap='Greys')

        # else:

            # ax[0, i-1].imshow(transfo_rgb(median_colors[np.where(counts==sorted_count[-i])[0][0]]/255))
            # a
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


def find_colors(flat_img, similarity, band_width, imshape, show=False):
    # image = clip_black(image, 0.2)
    # low_res = lower_resolution(image, 10)
    # flat_img = low_res.reshape(low_res.shape[0]*low_res.shape[1], 3)
    counts = [] #np.zeros((np.shape(flat_img)[0]))
    median_colors = []
    density_colors = []
    color_groups = []
    group_number = 0
    # pix_i = 0
    n_group_founds = 0
    # indices = 
    # loop until we have remove all the image pixels
    while(len(flat_img) > 0):

        list_sim = []
        # sum_color = [0, 0, 0]
        sim_colors = []
        counts.append(1)
        for j in range(1, len(flat_img)):
            ''' check if the distance between the first pixel of the list
            of remaining pixels and the rest of the list
            is lower than
            the accepted similarity'''

            # if np.sum(flat_img[j]-flat_img[pix_i]) < similarity:
            # print(flat_img[j].shape)
            # print(flat_img[pix_i].shape)
            # print(j, flat_img.shape)
            distance = np.sqrt((flat_img[j, 0] - flat_img[0, 0])**2 +
                                (flat_img[j, 1] - flat_img[0, 1] )**2 +
                                (flat_img[j, 2] - flat_img[0, 2])**2)
            
            if distance < similarity:# np.linalg.norm(flat_img[j]-flat_img[pix_i]) < similarity: 
                ''' if the pixel is the similar, we add it to the list
                of similar pixels '''
                list_sim.append(j)
                # sum_color += flat_img[j]
                # x, y = np.unravel_index(j, imshape)
                sim_colors.append(flat_img[j])
                counts[-1] += 1
    
        ''' after finding all pixels similar to pix_i, compute
        the median color ''' 
        # median_colors.append(np.array(sum_color) / len(list_sim))
        # print()
        color_groups.append(sim_colors)
        median_colors.append(np.median(sim_colors, axis=0))
        n_group_founds += 1
        ''' Delete all the found pixels, so that we can go to the
        next color still present in the painting '''
        flat_img = np.delete(flat_img, list_sim, 0)
        flat_img = np.delete(flat_img, 0, 0)

    sorted_indices = sorted(range(len(counts)),
                                    key=lambda i: counts[i], reverse=True)
    color_groups = sorted(color_groups, key=len, reverse=True)
    median_colors = [median_colors[i] for i in sorted_indices]
    return median_colors, color_groups, counts, density_colors
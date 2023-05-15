#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 19:04:29 2023

@author: borjangeshkovski
"""

import numpy as np
import matplotlib.pyplot as plt

def get_colormap(mean, cmap):
    
    maxima = [[i, mean[i][-1]] for i in range(nb_coords)]
    order = sorted(maxima, key=lambda x: x[1])
    
    indices = [x[0] for x in order]
    colors = plt.cm.get_cmap(cmap, nb_coords)
    color_list = [0 for i in range(nb_coords)]
    for i in range(nb_coords):
        color_list[indices[i]] = colors(i)
    return color_list

def plot_std(means, stds, file_name, title, cmap):
    
    plt.figure()
    plt.xlim(0, 15)

    # Remove the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    color_list = get_colormap(means, cmap)

    # Plot the arrays with varying colors
    for i in range(nb_coords):
        #plt.plot(np.linspace(0, 15, num_steps), means[i], color=color_list[i], linewidth=0.5)
        
        plt.fill_between(np.linspace(0, 15, num_steps), means[i] - stds[i], means[i] + stds[i],
                          color=color_list[i], alpha=0.2)
        
    plt.plot(np.linspace(0, 15, num_steps), means[0], color='white', linewidth=0.5)
    plt.plot(np.linspace(0, 15, num_steps), means[-1], color='white', linestyle=':', linewidth=0.75)

      
    plt.yscale('symlog')
    print(len(plt.gca().get_yticklabels()))
    plt.gca().set_yticklabels([r'$-10^{14}$', '', r'$-10^{10}$', '', r'$-10^{6}$', '', r'$-10^{2}$', '', r'$0$', '', r'$10^2$', '', r'$10^{6}$', '', r'$10^{10}$', '', r'$10^{14}$'])
    plt.xlabel(r'$t$', fontsize=14)
    
    plt.rcParams['text.usetex'] = True
    plt.title(title, fontsize=18)
    plt.savefig('%s.pdf' % file_name, format='pdf', bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":

    z = np.load('cupy_array.npy')
    V = np.load('cupy_array_V.npy')
        
    coordinates = [coord for coord, value in enumerate(z[1, -1, :]) if abs(value)>1]
    
    nb_coords = len(coordinates)
    num_steps = len(z[1, :, 1])
    n = 128
    
    mean = np.zeros((nb_coords, num_steps))
    std = np.zeros((nb_coords, num_steps))
    
    for i, coord in enumerate(coordinates):
        for tk in range(num_steps):
            
            mean[i, tk] = np.mean(z[:, tk, coord])
            std[i, tk] = np.std(z[:, tk, coord])
            
    title = r'Non-clustered coordinates'
    plot_std(mean, std, "stds_div", title, 'cividis')    

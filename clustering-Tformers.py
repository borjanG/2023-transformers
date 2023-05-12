#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

import numpy as np
import scipy as sp
import imageio
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from datetime import datetime


def get_dynamics(z_curr, attention, V, i):
    """
    - Returns: the dynamics z'(t) = (z_1'(t), ... , z_n'(t)) at some time-step t.
    """
    
    dlst = np.array([attention[i][j]*np.matmul(V, z_curr[j]-z_curr[i]) for j in range(n)])
    return np.sum(dlst, axis=0)

def transformer(T, dt, n, d, A, V, x0):
    """
    - Returns: the evolution of z = (z_1, ..., z_n) over time.
    """
    
    num_steps = int(T/dt)+1
    z = np.zeros(shape=(n, num_steps, d))
    z[:, 0, :] = x0
    integration_time = np.linspace(0, T, num_steps)

    for l, t in enumerate(integration_time):
        if l < num_steps - 1:
            # Attention matrix
            attention = [[1/np.sum([np.exp(np.dot(np.matmul(np.matmul(A, sp.linalg.expm(V*t)), z[i][l]), np.matmul(np.matmul(A, sp.linalg.expm(V*t)), z[k][l]-z[j][l]))) for k in range(n)]) for j in range(n)] for i in range(n)]
            
            z_next = np.zeros((n, d))
            for i in range(n):
                k1 = dt * get_dynamics(z[:, l, :], attention, V, i)
                k2 = dt * get_dynamics(z[:, l, :] + k1 / 2, attention, V, i)
                k3 = dt * get_dynamics(z[:, l, :] + k2 / 2, attention, V, i)
                k4 = dt * get_dynamics(z[:, l, :] + k3, attention, V, i)
                
                z_next[i] = z[i][l] + (k1 + 2*k2 + 2*k3 + k4) / 6
        
            z[:, l+1, :] = z_next
    return z

def calculate_distance_from_viewing_direction(point):
    viewing_direction = np.array([1, 0, 0])
    return np.linalg.norm(point - viewing_direction)

def visuals(d, 
            dt, 
            z, 
            integration_time, 
            conv, 
            color, 
            show_polytope, 
            movie, 
            dir_path, 
            base_filename):
    """
    Plots several snapshots of the trajectories {z_i(t)}_{i\in[n]}, and a movie thereof
    if indicated.
    """
    
    x_min, x_max = z[:, :, 0].min(), z[:, :, 0].max()
    if d>1:
        y_min, y_max = z[:, :, 1].min(), z[:, :, 1].max()
        if d == 3:
            z_min, z_max = z[:, :, 2].min(), z[:, :, 2].max()
    
    margin = 0.1
    x_range = x_max - x_min
    x_min -= margin * x_range
    x_max += margin * x_range

    if d>1:
        y_range = y_max - y_min
        y_min -= margin * y_range
        y_max += margin * y_range
        if d == 3:
            z_range = z_max - z_min
            z_min -= margin * z_range
            z_max += margin * z_range
    
    interp_x = []
    interp_y = []
    interp_z = []

    for i in range(n):
        interp_x.append(interp1d(integration_time, z[i, :, 0], 
                                 kind='cubic', 
                                 fill_value='extrapolate'))
        if d>1:
            interp_y.append(interp1d(integration_time, z[i, :, 1], 
                                     kind='cubic', 
                                     fill_value='extrapolate'))
            if d==3:
                interp_z.append(interp1d(integration_time, z[i, :, 2], 
                                         kind='cubic', 
                                         fill_value='extrapolate'))
                
    rc("text", usetex = True)
    font = {'size'   : 16}
    rc('font', **font)
    
    for t in range(num_steps):
        if d == 2 and round(t*dt % 0.5, 10) == 0:
            label_size = 16
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size 
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.title(r'$t={t}$'.format(t=str(round(t*dt, 2))), fontsize=20)
                
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.scatter([x(integration_time)[t] for x in interp_x], 
                        [y(integration_time)[t] for y in interp_y], 
                        c=color, 
                        alpha=1, 
                        marker = 'o', 
                        linewidth=0.75, 
                        edgecolors='black', 
                        zorder=3)
            
            plt.scatter([x(integration_time)[0] for x in interp_x], 
                        [y(integration_time)[0] for y in interp_y], 
                        c='white', 
                        alpha=0.1, 
                        marker = '.', 
                        linewidth=0.3, 
                        edgecolors='black', 
                        zorder=3)
            
            if t > 0:
                for i in range(n):
                    x_traj = interp_x[i](integration_time)[:t+1]
                    y_traj = interp_y[i](integration_time)[:t+1]
                    plt.plot(x_traj, 
                             y_traj, 
                             c=color, 
                             alpha=1, 
                             linewidth = 0.25, 
                             linestyle = 'dashed',
                             zorder=1)
            if conv:
                points = np.array([[z[i, t, 0], 
                                    z[i, t, 1]] for i in range(n)])
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], 
                             points[simplex, 1], 
                             color='silver',
                             linewidth = 0.5,
                             alpha=1)
            
            plt.savefig(base_filename + "{}.pdf".format(t), 
                        format='pdf', 
                        bbox_inches='tight')
                    
        elif d == 3 and round(t*dt % 0.5, 10) == 0:
            fig = plt.figure()
            ax = Axes3D(fig)
            label_size = 16
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size
            
            plt.title(r'$t={t}$'.format(t=str(round(t*dt,2))), fontsize=20)
                
            ax.scatter([x(integration_time)[t] for x in interp_x], 
                        [y(integration_time)[t] for y in interp_y],
                        [z(integration_time)[t] for z in interp_z],
                        c=color, 
                        alpha=1, 
                        marker = 'o', 
                        linewidth=0.75, 
                        edgecolors='black')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            
            ax.scatter([x(integration_time)[0] for x in interp_x], 
                        [y(integration_time)[0] for y in interp_y],
                        [z(integration_time)[0] for z in interp_z], 
                        c='white', 
                        alpha=0.1, 
                        marker = '.', 
                        linewidth=0.3, 
                        edgecolors='black', 
                        zorder=3)
            
            if t > 0:
                for i in range(n):
                    x_traj = interp_x[i](integration_time)[:t+1]
                    y_traj = interp_y[i](integration_time)[:t+1]
                    z_traj = interp_z[i](integration_time)[:t+1]
                    ax.plot(x_traj, 
                            y_traj, 
                            z_traj, 
                            c=color, 
                            alpha=0.75, 
                            linestyle = 'dashed',
                            linewidth = 0.25)
            
            ax.set_xlim3d(x_min, x_max)
            ax.set_ylim3d(y_min, y_max)
            ax.set_zlim3d(z_min, z_max)
            
            ax.view_init(elev=10)
        
            ax.grid(False)
            plt.locator_params(nbins=4)
            
            if conv:
                points = np.array([[z[i, t, 0], 
                                    z[i, t, 1],
                                    z[i, t, 2]] for i in range(n)])
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    plt.plot(points[simplex, 0], 
                             points[simplex, 1], 
                             points[simplex, 2],
                             color='silver',
                             linewidth = 0.5,
                             alpha=1)
            
            plt.savefig(os.path.join(dir_path, base_filename + "{}.pdf".format(t)), 
                        format='pdf', 
                        bbox_inches='tight')
            
            if t == num_steps-1 and show_polytope:
                
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                fig = plt.figure()
                ax = Axes3D(fig)
                plt.title("")
                fig.set_facecolor('white')
                ax.set_facecolor('white') 
                ax.grid(False) 
                ax.w_xaxis.pane.fill = False
                ax.w_yaxis.pane.fill = False
                ax.w_zaxis.pane.fill = False
                ax.set(xlabel='')

                # Remove y-axis label
                ax.set(ylabel='')
                ax.set_zlabel('')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                    
                ax.scatter([x(integration_time)[t] for x in interp_x], 
                            [y(integration_time)[t] for y in interp_y],
                            [z(integration_time)[t] for z in interp_z],
                            c=color, 
                            alpha=1, 
                            marker = 'o', 
                            linewidth=0.75, 
                            edgecolors='black')
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
                
                ax.scatter([x(integration_time)[0] for x in interp_x], 
                            [y(integration_time)[0] for y in interp_y],
                            [z(integration_time)[0] for z in interp_z], 
                            c='white', 
                            alpha=0.1, 
                            marker = '.', 
                            linewidth=0.3, 
                            edgecolors='black', 
                            zorder=3)
                
                
                points = np.array([[z[i, t, 0], 
                                    z[i, t, 1],
                                    z[i, t, 2]] for i in range(n)])
                hull = ConvexHull(points)
                
                cmap = plt.cm.get_cmap("plasma")
                
                for s in hull.simplices:
                    s = np.append(s, s[0])  # Here we cycle back to the first vertex in each simplex.
                    poly = Poly3DCollection([points[s]])
                    
                    min_x = np.min(points[s][:, 0])
                    color = cmap(min_x / 8)
            
                    # Calculate alpha based on the minimum X-coordinate, with front facets being more transparent.
                    alpha = 0.35 - 0.15 * (min_x / 8)
                    
                    poly.set_facecolor((*color[:3], alpha))  # Set the facecolor with the calculated color and alpha value.
                    ax.add_collection3d(poly)
                    
                for i in range(n):
                    x_traj = interp_x[i](integration_time)[:t+1]
                    y_traj = interp_y[i](integration_time)[:t+1]
                    z_traj = interp_z[i](integration_time)[:t+1]
                    ax.plot(x_traj, 
                            y_traj, 
                            z_traj, 
                            c='#FA7000', 
                            alpha=1, 
                            linestyle = 'dashed',
                            linewidth = 0.6)
                   
                ax.view_init(elev=10)
                
                plt.savefig(os.path.join(dir_path, base_filename + "{}.pdf".format(t+1)), 
                            format='pdf', 
                            bbox_inches='tight')
         
        if movie:           
            plt.savefig(base_filename + "{}.png".format(t),
                        format='png', dpi=250, bbox_inches=None)
        plt.clf()
        plt.close()
        
    if movie:
        imgs = []
        for i in range(num_steps):
            img_file = base_filename + "{}.png".format(i)
            imgs.append(imageio.imread(img_file))
            os.remove(img_file) 
        imageio.mimwrite(os.path.join(dir_path, filename), imgs)
 

geometries = ["polytope", "hyperplanes", "codimension-k", "hyperplanes x polytope"]
T = 5
dt = 0.1
n = 20
d = 2
x0 = np.random.uniform(low=-5, high=5, size=(n, d))
num_steps = int(T/dt)+1
integration_time = np.linspace(0, T, num_steps)

movie = False
conv = False
show_polytope = False

# An example
choice = "polytope"

if choice not in geometries:
    import sys
    print("Element is not in the list")
    sys.exit()
print("Element is in the list")

if choice == "polytope":
    # Theorem 3.2 (Clustering to a convex polytope)
    A = np.eye(d)
    V = np.eye(d)
    dir_path = './Th31'
    color = '#FA7000'
    conv = True

if choice == "hyperplanes":
    # Theorem 3.1 (Clustering to hyperplanes)
    A = np.eye(d)
    V = np.random.rand(d, d)
    V = np.abs(V)               # Perron Frobenius
    print("Eigenvalues of V:")
    print(np.linalg.eigvals(V))
    # Example in paper (2d):
    # V = np.array([[0.64709742, 0.81911926],
    #                0.61210449, 0.63263484]])
    # Example in paper (3d; Re(l)<0):
    # V = np.array([[0.2362413 , 0.0599536 , 0.40506019],
    #         [0.81321448, 0.49948705, 0.33754772],
    #         [0.2160645 , 0.38505272, 0.14588075]])
    # # Example 2 in paper (3d; Re(l)>0):
    # V = np.array([[0.18608476, 0.34391026, 0.13905949],
    #         [0.35244288, 0.6694859 , 0.39992145],
    #         [0.46685606, 0.17094454, 0.42092308]])
    dir_path = './Th41'
    color = '#3a4cc1'

if choice == "codimension-k":
    # Conjecture (codimension-k)
    A = np.eye(d)
    V = np.random.rand(d, d)
    V = (V+V.T)/2
    print("Eigenvalues of V:")
    print(np.linalg.eigvals(V))
    # Example in paper: Real eigenvalues (2 positive, 1 negative)
    # V = np.array([[0.92761287, 0.94333073, 0.3827744 ],
    #         [0.94333073, 0.59991396, 0.47825472],
    #         [0.3827744 , 0.47825472, 0.46668605]])
    dir_path = './Conj'
    color = '#9f4292'

if choice == "hyperplanes x polytope":
    # Theorem 5.1 (Clustering to a polytope and hyperplane)
    A = np.eye(d)
    # Example in paper:
    V = np.eye(d)
    V[-1][-1] = -0.5
    dir_path = './Th51'
    color = '#63acbe'
    conv = True
    
# The trajectories
z = transformer(T, dt, n, d, A, V, x0)

# Set the directory path where you want to save the pictures
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Generate the filename using the current datetime
now = datetime.now()
dt_string = now.strftime("%H-%M-%S")
filename = dt_string + ".gif"
base_filename = dt_string 
    
visuals(d, dt, z, integration_time, conv, color, show_polytope, movie, dir_path, base_filename)



            
        

        

    





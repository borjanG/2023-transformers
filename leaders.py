#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

import numpy as np
import matplotlib.pyplot as plt

n = 20
T = 10
dt = 0.1
num_steps = int(T/dt)+1
d = 2

V = 2*np.random.rand(d, d)-np.ones((d, d))
V = np.matmul(V, V.T)
A = 2*np.random.rand(d, d)-np.ones((d, d))
A = np.matmul(A, A.T)

x0 = np.random.uniform(low=-1, high=1, size=(n, d))
x = np.zeros(shape=(n, num_steps, d))
x[:, 0, :] = x0
integration_time = np.linspace(0, T, num_steps)

for l, t in enumerate(integration_time):
    if l<num_steps-1:
        # Attention matrix
        attention = [[1/np.sum([np.exp(np.dot(np.matmul(A, x[i][l]), x[k][l]-x[j][l])) for k in range(n)]) for j in range(n)] for i in range(n)]
        print(np.linalg.matrix_rank(attention))
        
        label_size = 0
        if round(t % 0.5, 10) == 0:
            
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            label_size = 8
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size 
            
            ax.set_aspect('equal', adjustable='box')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.scatter([np.matmul(A, x[i,l])[0] for i in range(n)],
                        [np.matmul(A, x[i,l])[1] for i in range(n)],
                        c="#a8deb5", 
                        alpha=1, 
                        marker = 'o', 
                        linewidth=0.75, 
                        edgecolors='black', 
                        zorder=3)
            
            plt.scatter([x[i,l][0] for i in range(n)],
                        [x[i,l][1] for i in range(n)],
                        c="#d91c72", 
                        alpha=1, 
                        marker = 'o', 
                        linewidth=0.75, 
                        edgecolors='black', 
                        zorder=3)
            
            plt.title(r'$t={t}$'.format(t=str(round(t, 2))), fontsize=20)
            
            for i in range(n):
                for j in range(n):
                    if attention[i][j]>1e-4 and i!=j:
                        plt.plot([np.matmul(A, x[i,l])[0], x[j,l][0]], 
                                  [np.matmul(A, x[i,l])[1], x[j,l][1]],
                                  linewidth=attention[i][j]*1e-6,
                                  color="black")
            
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            
            plt.savefig("connect-{}.pdf".format(round(t, 2)), 
                        format='pdf', 
                        bbox_inches='tight')
        
        for i in range(n):
            dlst = np.array([attention[i][j]*np.matmul(V, x[j][l]) for j in range(n)])
            # sum over j
            dynamics = np.sum(dlst, axis=0)
            # Euler scheme
            x[i][l+1] = x[i][l] + dt*dynamics 
        

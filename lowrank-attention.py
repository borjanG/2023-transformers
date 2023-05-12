#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

import numpy as np
import matplotlib.pyplot as plt

n = 40
T = 11
dt = 0.1
num_steps = int(T/dt)+1
d = 80

A = np.eye(d)
V = 2*np.random.rand(d, d)-np.ones((d, d))
V = np.matmul(V, V.T)

x0 = np.random.uniform(low=-1, high=1, size=(n, d))
x = np.zeros(shape=(n, num_steps, d))
x[:, 0, :] = x0
integration_time = np.linspace(0, T, num_steps)

for l, t in enumerate(integration_time):
    if l<num_steps-1:
        attention = [[1/np.sum([np.exp(np.dot(np.matmul(A, x[i][l]), x[k][l]-x[j][l])) for k in range(n)]) for j in range(n)] for i in range(n)]
        print(np.linalg.matrix_rank(attention))
        
        label_size = 0
        plt.matshow(attention, cmap="Blues")
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        plt.title(r'$t={t}$, rank$={r}$'.format(t=str(round(t, 2)), r=str(np.linalg.matrix_rank(attention))), fontsize=20)
        plt.savefig("attention-{}.pdf".format(round(t, 2)),
                    format='pdf', bbox_inches=None, rasterized=True) 
            
        plt.clf()
        plt.close()
            
        for i in range(n):
            k1 = dt*np.matmul(V, x[i][l])
            k2 = dt*np.matmul(V, x[i][l] + 0.5*k1)
            k3 = dt*np.matmul(V, x[i][l] + 0.5*k2)
            k4 = dt*np.matmul(V, x[i][l] + k3)
            dynamics = (1/6)*(k1 + 2*k2 + 2*k3 + k4)
            x[i][l+1] = x[i][l] + dynamics
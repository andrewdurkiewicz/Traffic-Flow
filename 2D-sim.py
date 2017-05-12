#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import gaussian_kde as gkde
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import spline
from scipy.stats import gaussian_kde


plt.ion()
T=400
tau = 1e-2
dt = 1e-3
N = 150
d = 2
x_total = 500
b = .9
v0_avg = 100
v0_variance = 10
r = (x_total/(2*np.pi))
n_bins = 50
t_range = np.arange(0,T+1,dt)


#Initialize our conditions
v = np.zeros(N)
x = np.zeros(N)
x_prev = np.zeros(N)
v_prev = np.zeros(N)
# v = np.zeros([len(t_range),N])
# x = np.zeros([len(t_range),N])
x_circle = np.zeros([len(t_range),N])
y_circle = np.zeros([len(t_range),N])
r_matrix = r* np.ones_like(x)


#set initial parameters
v_prev[:] = (v0_avg-v0_variance) + v0_variance*np.random.rand(N)
x_prev[:] = np.sort(random.sample(range(x_total), N))

#calculate the new velocity and position
for t in range(len(t_range)-1):
    # for i in range(N):
    #     dx = -1*(x[t,i-1]-x[t,i])
    #     dx += x_total*(dx < 0)
    #     v[t+1,i] = dt * b*(dx - tau*v[t,i])+v[t,i]
    #     x[t+1,i] = (x[t,i] + dt * (v[t,i]+v[t+1,i])/2)%x_total
    x_trans = np.append(np.append([x_prev[-1]],x_prev[:]),[x_prev[0]])
    dx = x_trans[1:-1] - x_trans[:-2]
    dx += x_total * (dx < 0)
    v[:] = dt * b * (dx - tau*v_prev[:]) + v_prev[:]
    x[:] = (x_trans[1:-1] + dt * (v_prev[:] + v[:])/2) % x_total
    plt.clf()
    # plt.hist(x[:], bins=25)
    plt.scatter( x[:], np.ones_like(x[:]),color = 'red',marker = 's')
    plt.xlim(0, x_total)
    plt.ylim(0,12)
    plt.title('Vehicle Density With N = %i, T = %-.3fs'%(N,t*dt))
    plt.xlabel('Road Position')
    plt.ylabel('Number Density')
    bin_range = np.linspace(0  , x_total, n_bins+1)
    bin_fine_range = np.linspace(0, x_total-x_total/n_bins, n_bins*4)+x_total/(2*n_bins)
    bin_values = [sum((bin_range[i] <= x) & (x <= bin_range[i+1])) for i in range(n_bins)]
    #plt.plot(bin_range[:-1], bin_values)
    plt.plot(bin_fine_range,spline(bin_range[:-1],bin_values,bin_fine_range),label = 'Spline Interpolation')

    plt.pause(.0001)
    plt.show()
    if t!=0 and 1/t > 1 and t%100 == 0:
        plt.savefig('Vehicle-Density-With-N-%i_T-%-.3fs.png'%(N,t*dt))
    x_prev[:], v_prev[:] = x[:], v[:]






# theta_matrix = x/r
# for m in range(N):
# 	y_circle[:,m] = r*np.cos(x[:,m]*2*np.pi/x_total)
# 	x_circle[:,m] = r*np.sin(x[:,m]*2*np.pi/x_total)
# plt.ion()
# fig = plt.figure()
# plt.autoscale(False)
# for k in range(0,T,3):

#     plt.clf()
#     ax = fig.add_subplot(1,2,1)
#     bx = fig.add_subplot(1,2,2,polar = True)
#     heatmap, xedges, yedges = np.histogram2d(x_circle[k,:],
#                                                 y_circle[k,:], bins=20)
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#     ax.imshow(heatmap.T, extent=extent, cmap = 'hot',origin='lower',
#                                     interpolation='gaussian', vmin=0,
#                                     vmax=1)
#     #for m in range(N):
#     bx.scatter(np.rad2deg(theta_matrix[k,:]),r_matrix[k,:], marker = 's',
#                                         edgecolor = 'black',linewidth = '1',
#                                          c = 'blue')
#     bx.grid(False)
#     bx.set_rmax(r)
#     xT = plt.xticks()[0]
#     xL =['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
#     			r'$\pi$',r'$\frac{5\pi}{4}$',
#     				r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']

#     ax.set_title(k)
#     ax.set_xlim(-r-5,r+5)
#     ax.set_ylim(-r-5,r+5)
#     fig.canvas.draw()

#     plt.hist(x[k, :], bins=25)
#     # plt.scatter( x[k, :], np.zeros_like(x[k, :]))
#     plt.xlim(0, x_total)
#     plt.ylim(0,4)
#     plt.title('T = %i'%k)
#     plt.pause(.0001)
#     plt.show()

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import gaussian_kde as gkde
from mpl_toolkits.mplot3d import Axes3D

T=30

tou = 5
dt = 1e-2
N = 50
d = 2
x_total = 300
b = .1
v0_avg = 40
v0_variance = 10
t_range = np.arange(0,T+1,dt)

v = np.zeros([len(t_range),N])
x = np.zeros([len(t_range),N])
avgdis = np.zeros([len(t_range),N])
x_circle = np.zeros([len(t_range),N])
y_circle = np.zeros([len(t_range),N])
#get coordinates for circle

v[0] = (v0_avg-v0_variance) + v0_variance*np.random.rand(N)
x[0] = np.sort(random.sample(range(x_total), N))
#x[1] = (x[0] + v[0]* tou)%x_total

plt.ion()
for t in range(len(t_range)-1):
    for i in range(N):
        dx = -1*(x[t,i-1]-x[t,i])
        dx += x_total*(dx < 0)
#         x[t+1,i] = tou**2 * b * (dx - d) - x[t-1,i] + 2*x[t,i]
#         x[t+1,i]%=x_total
#         assert 0 <= (dx + x_total*(dx < 0))*
        v[t+1,i] = dt * b*(dx - tou*v[t,i])+v[t,i]
        x[t+1,i] = (x[t,i] + dt * (v[t,i]+v[t+1,i])/2)%x_total
r = (x_total/(2*np.pi))
r_matrix = r* np.ones_like(x)
theta_matrix = x/r
for m in range(N):
	y_circle[:,m] = r*np.cos(x[:,m]*2*np.pi/x_total)
	x_circle[:,m] = r*np.sin(x[:,m]*2*np.pi/x_total)


for k in range(len(t_range)-1):
	plt.clf()
	plt.subplot(111, projection='polar')

	# ax = fig.add_subplot(111)
	# # bx = fig.add_subplot(111,polar = True)
	# heatmap, xedges, yedges = np.histogram2d(x_circle[k,:], y_circle[k,:], bins=20)
	# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

	#cx = ax.imshow(heatmap.T, extent=extent, cmap = 'magma',origin='lower',interpolation='gaussian', vmin=0, vmax=1 )
	for m in range(N-1):
		plt.scatter(np.rad2deg(theta_matrix[k,m]),r_matrix[k,m], marker = 's',
						edgecolor = 'black',linewidth = '1', c = 'blue')
	for m in range(N-1,N):
		plt.scatter(np.rad2deg(theta_matrix[k,m]),r_matrix[k,m], marker = 's',
						edgecolor = 'black',linewidth = '1', c = 'red')


	xT=plt.xticks()[0]
	xL=['0',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',\
	    r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
	plt.xticks(xT, xL)
	# cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])  # This is the position for the colorbar
	# cb = plt.colorbar(cx, cax = cbaxes)	
	# axis = r + 20
	# ax.set_xlim(-axis,axis)
	# ax.set_ylim(-axis,axis)
	# plt.colorbar(cx)
	# plt.xlabel('X Position')
	# plt.ylabel('Y Position')
	plt.title('Vehicle Density With N = %i, T = %-.2fs'%(N,k*dt))

	plt.pause(.001)
	plt.savefig("n-%i_xmax-%i-image%03d"".jpeg" %(N,x_total,k))

	plt.show()
#return v,N,t_range,T

# v_soln,N,tx,t_total = soln()

# #plt.plot(tx,v_soln[0,:])
# # for m in range(N):
# # 	for k in range(len(tx)):
# # 		plt.plot(tx,v_soln[k,:])
# # plt.show()
# for m in range(0,10,2):
# 	plt.plot(tx,v_soln[:,m],label = 'Vehicle %i'%(m+1))

# plt.title('Velocity Sampling vs Time')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Velocity (mph)')
# plt.legend()
# plt.savefig('v-vs-t.png')
# plt.show()
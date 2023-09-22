from math import pi
from matplotlib import pyplot as plt
import numpy as np
from util_2d import *
plt.rcParams.update({'font.size': 16})

#first we initialize some parameters.

N = 20 #number of grid points, i.e. num of states.
kT = 0.5981
t_max = 10**7 #max time
ts = 0.1 #time step

x,y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N))
state_start = (14, 14)
state_end = (4, 6)

#map it into xy mesh.

img = Image.open("./fes_digitize.png")
img = np.array(img)

img_greyscale = 0.8 * img[:,:,0] - 0.15 * img[:,:,1] - 0.2 * img[:,:,2]
img = img_greyscale
img = img/np.max(img)
img = img - np.min(img)

#get img square and multiply the amp = 7
min_dim = min(img.shape)
img = img[:min_dim, :min_dim]
img = 7 * img

plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
plt.savefig("./figs/unbiased.png", dpi=600)
plt.show()


traj_all = np.load("./data/CV_total_20230921-114158_2.npy") #note this is ravelled.
traj_1 = traj_all[:3001]
traj_2 = traj_all[3001:6002]
traj_3 = traj_all[6002:9003]

#unravel the traj.
traj_1 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_1])
traj_2 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_2])
traj_3 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_3])

#load gaussian parameters
gp1 = np.load("./data/gaussian_params_20230921-114158_1.npy")
gp2 = np.load("./data/gaussian_params_20230921-114158_2.npy")


#first we plot the unbiased FES and its traj.

traj_1_x_indices = traj_1[:, 0]
traj_1_y_indices = traj_1[:, 1]
traj_1_x_coords = x[traj_1_x_indices, traj_1_y_indices]
traj_1_y_coords = y[traj_1_x_indices, traj_1_y_indices]

plt.figure()
plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
plt.plot(x[state_start], -y[state_start], marker = 'o', color = "red", markersize = 10) #this is starting point.
plt.plot(x[state_end], -y[state_end],marker = 'x', color = "red", markersize = 10) #this is ending point.
plt.plot(traj_1_x_coords, -traj_1_y_coords, color="yellow", linewidth=0.8, alpha = 0.8,)
#plt.title("")
plt.savefig("./figs/prod_figs/traj1.png")
#plt.show()
plt.close()

#next we plot the biased FES after traj1.
from util_2d import *
x_img, y_img = np.meshgrid(np.linspace(-3,3,min_dim), np.linspace(-3,3,min_dim))
total_bias = get_total_bias_2d(x_img, y_img, gp1)

#add the bias to the unbiased FES.
img_biased = img + total_bias

#process traj_2

traj_2_x_indices = traj_2[:, 0]
traj_2_y_indices = traj_2[:, 1]
traj_2_x_coords = x[traj_2_x_indices, traj_2_y_indices]
traj_2_y_coords = y[traj_2_x_indices, traj_2_y_indices]

plt.figure()
plt.imshow(img_biased, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
plt.plot(x[state_start], -y[state_start], marker = 'o', color = "red", markersize = 10) #this is starting point.
plt.plot(x[state_end], -y[state_end],marker = 'x', color = "red", markersize = 10) #this is ending point.
plt.plot(traj_1_x_coords, -traj_1_y_coords, color="yellow", linewidth=0.8, alpha = 0.4,)
plt.plot(traj_2_x_coords, -traj_2_y_coords, color="yellow", linewidth=0.8, alpha = 0.8,)
#plot the last position of traj_2
plt.plot(traj_2_x_coords[-1], -traj_2_y_coords[-1], marker = 'o', color = "red", markersize = 10)
#plt.show()
plt.savefig("./figs/prod_figs/traj2.png")
plt.close()


#next we plot the biased FES after traj2.
total_bias = get_total_bias_2d(x_img, y_img, gp2)
img_biased = img + total_bias
#process traj_3
traj_3_x_indices = traj_3[:, 0]
traj_3_y_indices = traj_3[:, 1]
traj_3_x_coords = x[traj_3_x_indices, traj_3_y_indices]
traj_3_y_coords = y[traj_3_x_indices, traj_3_y_indices]

plt.imshow(img_biased, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
plt.plot(x[state_start], -y[state_start], marker = 'o', color = "red", markersize = 10) #this is starting point.
plt.plot(x[state_end], -y[state_end],marker = 'x', color = "red", markersize = 10) #this is ending point.
plt.plot(traj_1_x_coords, -traj_1_y_coords, color="yellow", linewidth=0.8, alpha = 0.4,)
plt.plot(traj_2_x_coords, -traj_2_y_coords, color="yellow", linewidth=0.8, alpha = 0.4,)
plt.plot(traj_3_x_coords, -traj_3_y_coords, color="yellow", linewidth=0.8, alpha = 0.8,)
plt.plot(traj_3_x_coords[-1], -traj_3_y_coords[-1], marker = 'o', color = "red", markersize = 10)
#plt.show()
plt.savefig("./figs/prod_figs/traj3.png")
plt.close()








print("done")
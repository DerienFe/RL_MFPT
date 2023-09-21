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
plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3])
plt.savefig("./figs/unbiased.png", dpi=600)
plt.show()


traj1 = np.load("./data/CV_total_20230920-125013_0.npy") #note this is ravelled.

traj1_processed = np.zeros((traj1.shape[0],2)) #for x,y
#we unravel the index, then map it to x,y.
for i in range(traj1.shape[0]):
    traj1_processed[i,0], traj1_processed[i,1] = np.unravel_index(int(traj1[i]), (N,N), order='C')
    traj1_processed[i,0] = x[0][int(traj1_processed[i,0])]
    traj1_processed[i,1] = y[int(traj1_processed[i,1])][0]

    traj1_processed[i,0], traj1_processed[i,1] = traj1_processed[i,1], traj1_processed[i,0]
       
    #flip the y
    traj1_processed[i,1] = - traj1_processed[i,1]



plt.figure()
plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3])
plt.plot(y[state_start[1]][0], -x[0][state_start[0]], marker = 'o', color = "red", markersize = 10) #this is starting point.
plt.plot(y[state_end[1]][0], -x[0][state_end[0]], marker = 'x', color = "red", markersize = 10) #this is ending point.
plt.scatter(traj1_processed[:,0], traj1_processed[:,1], color="yellow", linewidth=0.5, alpha = 0.3, s = 15)
plt.title("")
plt.savefig("./figs/prod_figs/traj1.png")
plt.show()


#now we plot the traj.

traj2 = np.load("./data/CV_total_20230920-130859_1.npy") #note this is ravelled.

traj2_processed = np.zeros((traj2.shape[0],2)) #for x,y
#we unravel the index, then map it to x,y.
for i in range(traj2.shape[0]):
    traj2_processed[i,0], traj2_processed[i,1] = np.unravel_index(int(traj2[i]), (N,N), order='C')
    traj2_processed[i,0] = x[0][int(traj2_processed[i,0])]
    traj2_processed[i,1] = y[int(traj2_processed[i,1])][0]
    
    #transpose the x and y
    traj2_processed[i,0], traj2_processed[i,1] = traj2_processed[i,1], traj2_processed[i,0]
        
    #flip the y
    traj2_processed[i,1] = - traj2_processed[i,1]



plt.figure()
plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3])
plt.plot(y[state_start[1]][0], -x[0][state_start[0]],  marker = 'o', color = "red", markersize = 10) #this is starting point.
plt.plot(y[state_end[1]][0], -x[0][state_end[0]], marker = 'x', color = "red", markersize = 10) #this is ending point.
plt.scatter(traj2_processed[:,0], traj2_processed[:,1], color="yellow", linewidth=0.5, alpha = 0.3, s = 15)
plt.title("")
plt.savefig("./figs/prod_figs/traj2.png")
plt.show()



print("done")
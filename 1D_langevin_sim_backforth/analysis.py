#post analysis code in python 15th Jan 2024
# by Tiejun Wei

import numpy as np
import matplotlib.pyplot as plt
import os
import natsort

#data path
traj_path = './visited_states/20231123-170259_pos_traj.txt' #'./visited_states/20240118-115232_pos_traj.txt'
bias_path = []

all_bias_path = os.listdir("./params/")
for path in all_bias_path:
    if path.startswith("20231123-170259"):      #("20240118-115232"):  
        bias_path.append("./params/"+path)
bias_path = natsort.natsorted(bias_path)

#load data
num_sim = 56
traj = np.loadtxt(traj_path)
#we took out 0 values in the traj
traj = traj[traj[:]!=0]
traj = traj.reshape([num_sim, 5000, 1])

bias = np.zeros([num_sim, 10, 3])
for i in range(num_sim):
    bias[i] = np.loadtxt(bias_path[i]).reshape(10,3, order='F') 

#unbias method
from wham import WHAM
from dham import DHAM

dham_bias = True
wham_bias = True

if wham_bias:
    w = WHAM()
    w.setup(traj, T=300, global_gaussian_params=bias)
    w.converge(0.0001)
    w.project_1d([1], 100)
    wham_space = w.projection_bins
    wham_fes = w.profile
    wham_fes = wham_fes - np.min(wham_fes)
    plt.figure()
    plt.plot(wham_space, wham_fes, label="WHAM")
    #plt.savefig("wham.png")
    print("wham done. ")
if dham_bias:
    d = DHAM(global_gaussian_params=bias, num_bins=100)
    d.setup(traj, T=300, prop_index=0, time_tag='test')
    d.lagtime = 1
    dham_qspace, mU, M = d.run(biased = True, plot = False, use_symmetry = True, use_dynamic_bins = False)
    mU = mU - np.min(mU)

    plt.plot(dham_qspace, mU, label="DHAM")
    print("dham done. ")

#here we get the original 1D fes.

def draw_fes(qspace):
    #load the original fes
    Z = np.zeros_like(qspace)
    k = 5
    max_barrier = '1e3'
    offset = 0.4

    num_hills = 9
    A_i = np.array([0.9, 0.3, 0.7, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * 4 #this is in kcal/mol.
    x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 4.75, 6, 1] # this is in nm.
    sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]

    x = qspace
    for i in range(num_hills):
        Z += A_i[i] * np.exp(-(x-x0_i[i])**2/(2*sigma_x_i[i]**2))

    Z += float(max_barrier)/4.184 * (1 / (1 + np.exp(k * (x - (-offset))))) #left
    Z += float(max_barrier)/4.184 * (1 / (1 + np.exp(-k * (x - (2 * np.pi + offset))))) #right

    Z = Z - np.min(Z)
    return Z
    

qspace = np.linspace(0, 2*np.pi, 100)
fes = draw_fes(qspace)
plt.plot(qspace, fes, label="original")
plt.legend()
plt.savefig("fes.png")
plt.close()
print("all done.")
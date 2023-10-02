#this is a langevin simulator in OPENMM.
# we put a particle in a box and simulate it with Langevin dynamics.
# the external force is defined using a function digitizing the phi/psi fes of dialanine.

import numpy as np
import matplotlib.pyplot as plt

import time

import openmm
from openmm import unit
from openmm.app.topology import Topology
from openmm.app.element import Element

from openmm.unit import Quantity
from openmm import Vec3
#first we initialize the system.
# topology

def apply_fes(system, particle_idx, gaussian_param, pbc = False, name = "FES"):
    """
    tether a particle given gaussian parameters
    """
    #unpack gaussian parameters
    num_gaussians = int(len(gaussian_param)/5)
    A = gaussian_param[:num_gaussians] *7 
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    y0 = gaussian_param[2*num_gaussians:3*num_gaussians]
    sigma_x = gaussian_param[3*num_gaussians:4*num_gaussians]
    sigma_y = gaussian_param[4*num_gaussians:5*num_gaussians]    
    

    #now we add the force for all gaussianss.
    
    energy = "0"
    force = openmm.CustomExternalForce(energy)
    for i in range(num_gaussians):
        if pbc:
            energy = f"A{i}*exp(-periodicdistance(x,0,0, x0{i},0,0)^2/(2*sigma_x{i}^2) - periodicdistance(0,y,0, 0,y0{i},0)^2/(2*sigma_y{i}^2))"
            force = openmm.CustomExternalForce(energy)
        else:
            energy = f"A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2) - (y-y0{i})^2/(2*sigma_y{i}^2))"
            force = openmm.CustomExternalForce(energy)

        #examine the current energy term within force.

        print(force.getEnergyFunction())

        force.addGlobalParameter(f"A{i}", A[i])
        force.addGlobalParameter(f"x0{i}", x0[i])
        force.addGlobalParameter(f"y0{i}", y0[i])
        force.addGlobalParameter(f"sigma_x{i}", sigma_x[i])
        force.addGlobalParameter(f"sigma_y{i}", sigma_y[i])
        force.addParticle(particle_idx)
        #we append the force to the system.
        system.addForce(force)

    return system


sim_steps = int(1e4)
pbc = True
time_tag = time.strftime("%Y%m%d-%H%M%S")

elem = Element(0, "X", "X", 1.0)
top = Topology()
top.addChain()
top.addResidue("xxx", top._chains[0])
top.addAtom("X", elem, top._chains[0]._residues[0])

mass = 12.0 * unit.amu
#starting point as [1.29,-1.29,0.0]
start = Quantity(value = [Vec3(2.12,2.12,0.0)], unit = unit.nanometers)
system = openmm.System()
system.addParticle(mass)

#potential setup
#first we load the gaussians from the file.
# params comes in A, x0, y0, sigma_x, sigma_y format.

gaussian_param = np.loadtxt("./fes_digitize_gauss_params.txt") 
n_gaussians = int(len(gaussian_param)/5)

system = apply_fes(system = system, particle_idx=0, gaussian_param = gaussian_param, pbc = pbc, name = "FES")
z_pot = openmm.CustomExternalForce("100000 * z^2") # very large force constant in z
z_pot.addParticle(0)
system.addForce(z_pot) #on z, large barrier

#pbc section
if pbc:
    a = unit.Quantity((np.pi*unit.nanometers, 0*unit.nanometers, 0*unit.nanometers))
    b = unit.Quantity((0*unit.nanometers, np.pi*unit.nanometers, 0*unit.nanometers))
    c = unit.Quantity((0*unit.nanometers, 0*unit.nanometers, 1*unit.nanometers)) # atom not moving in z so we set it to 1 nm
    system.setDefaultPeriodicBoxVectors(a,b,c)




#integrator

integrator = openmm.LangevinIntegrator(300*unit.kelvin, 
                                       1.0/unit.picoseconds, 
                                       0.002*unit.picoseconds)

#run the simulation
simulation = openmm.app.Simulation(top, system, integrator, openmm.Platform.getPlatformByName('CUDA'), pbc = pbc)
simulation.context.setPositions(start)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)


pos_traj = np.zeros([sim_steps, 3])
for i in range(sim_steps):
    simulation.step(1)
    state = simulation.context.getState(getPositions=True)
    pos_traj[i,:] = state.getPositions(asNumpy=True)[0,:]


### VISUALIZATION ###

#plot the trajectory over -pi, pi range.
# also plot the fes. loaded in img.
# use scatter plot to plot the traj
from PIL import Image
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
plt.figure()
plt.imshow(img, cmap="coolwarm", extent=[-np.pi, np.pi,-np.pi, np.pi], vmin=0, vmax=12)
plt.scatter(pos_traj[:,0], pos_traj[:,1], s=0.5, alpha=0.5, c='yellow')
plt.xlabel("x")
plt.xlim([-np.pi - 1, np.pi+1])
plt.ylim([-np.pi-1, np.pi+1])
plt.ylabel("y")
plt.title("Trajectory")
plt.show()



#save the plot and the trajectory
np.savetxt(f"./langevin_approach/traj/traj_{time_tag}.txt", pos_traj)
plt.savefig(f"./langevin_approach/figs/traj_{time_tag}.png")

print("all done")

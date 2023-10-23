#this is a langevin simulator in OPENMM.
# we put a particle in a box and simulate it with Langevin dynamics.
# the external force is defined using a function digitizing the phi/psi fes of dialanine.

import numpy as np
import matplotlib.pyplot as plt

import time

from tqdm import tqdm

import openmm
from openmm import unit
from openmm.app.topology import Topology
from openmm.app.element import Element

from openmm.unit import Quantity
from openmm import Vec3

import config

from util import *

#first we initialize the system.
# topology

#for amp applied on fes. note the gaussian parameters for fes is normalized.

elem = Element(0, "X", "X", 1.0)
top = Topology()
top.addChain()
top.addResidue("xxx", top._chains[0])
top.addAtom("X", elem, top._chains[0]._residues[0])

mass = 12.0 * unit.amu
#starting point as [1.29,-1.29,0.0]
system = openmm.System()
system.addParticle(mass)

#potential setup
#first we load the gaussians from the file.
# params comes in A, x0, y0, sigma_x, sigma_y format.

gaussian_param = np.loadtxt("./params/gaussian_fes_param.txt") 

system, fes = apply_fes(system = system, 
                   particle_idx=0, 
                   gaussian_param = gaussian_param, 
                   pbc = config.pbc, 
                   name = "FES", 
                   amp=config.amp, 
                   mode = config.fes_mode,
                   plot = True)
z_pot = openmm.CustomExternalForce("100000 * z^2") # very large force constant in z
z_pot.addParticle(0)
system.addForce(z_pot) #on z, large barrier

#pbc section
if config.pbc:
    a = unit.Quantity((2*np.pi*unit.nanometers, 0*unit.nanometers, 0*unit.nanometers))
    b = unit.Quantity((0*unit.nanometers, 2*np.pi*unit.nanometers, 0*unit.nanometers))
    c = unit.Quantity((0*unit.nanometers, 0*unit.nanometers, 1*unit.nanometers)) # atom not moving in z so we set it to 1 nm
    system.setDefaultPeriodicBoxVectors(a,b,c)


#integrator
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 
                                       1.0/unit.picoseconds, 
                                       0.002*unit.picoseconds)

#before run, last check pbc:
#print(system.getDefaultPeriodicBoxVectors())

#CUDA
platform = openmm.Platform.getPlatformByName('CUDA')

#run the simulation
simulation = openmm.app.Simulation(top, system, integrator, platform)
simulation.context.setPositions(config.start_state)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
if config.pbc:
    simulation.context.setPeriodicBoxVectors(a,b,c)


pos_traj = np.zeros([config.sim_steps, 3])
for i in tqdm(range(config.sim_steps)):
    simulation.step(1)
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=config.pbc)
    pos_traj[i,:] = state.getPositions(asNumpy=True)[0,:]


### VISUALIZATION ###
x,y = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, 2*np.pi, 100)) #fes in shape [100,100]

plt.figure()
plt.imshow(fes, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=config.amp * 12/7, origin = "lower")
plt.scatter(pos_traj[::3,0], pos_traj[::3,1], s=0.5, alpha=0.5, c='yellow')
plt.xlabel("x")
plt.xlim([-1, 2*np.pi+1])
plt.ylim([-1, 2*np.pi+1])
plt.ylabel("y")
plt.title(f"Unbiased Trajectory, pbc={config.pbc}")
plt.colorbar()
#plt.show()
plt.savefig(f"./figs/unbias_traj_{config.time_tag}.png")
plt.close()



#save the plot and the trajectory
np.savetxt(f"./langevin_approach/traj/traj_{config.time_tag}.txt", pos_traj)

print("all done")

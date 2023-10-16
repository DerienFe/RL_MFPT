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
#first we initialize the system.
# topology

def apply_fes(system, particle_idx, gaussian_param, pbc = False, name = "FES", amp = 7):
    """
    this function apply the bias given by the gaussian_param to the system.
    paramerter coming pattern: A, x0, y0, sigma_x, sigma_y params[i*5:i*5+5]
    """
    #unpack gaussian parameters
    num_gaussians = int(len(gaussian_param)/5)
    A = gaussian_param[0::5] * amp
    x0 = gaussian_param[1::5]
    #we flip the yaxis, so we change the y0. y0' = 2pi - y0
    y0 = 2*np.pi - gaussian_param[2::5]
    sigma_x = gaussian_param[3::5]
    sigma_y = gaussian_param[4::5]

    #now we add the force for all gaussians.
    #in openmm energy always in kj/mol.
    energy = "0" 
    force = openmm.CustomExternalForce(energy)
    for i in range(num_gaussians):
        if pbc:
            energy = f"A{i}*exp(-(periodicdistance(x,0,0, x0{i},0,0)^2/(2*sigma_x{i}^2) + periodicdistance(0,y,0, 0,y0{i},0)^2/(2*sigma_y{i}^2)))"
            force = openmm.CustomExternalForce(energy)
        else:
            energy = f"A{i}*exp(-((x-x0{i})^2/(2*sigma_x{i}^2) + (y-y0{i})^2/(2*sigma_y{i}^2)))"
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
def sum_of_gaussians(params, x, y, n_gaussians, N=100):
    total = np.zeros((N,N))
    
    A = params[0::5]
    x0 = params[1::5]
    y0 = params[2::5]
    sigma_x = params[3::5]
    sigma_y = params[4::5]

    for i in range(n_gaussians):
        total += gaussian_2D([A[i], x0[i], y0[i], sigma_x[i], sigma_y[i]], x, y)
    return total
def gaussian_2D(params, x, y):
    A, x0, y0, sigma_x, sigma_y = params
    return A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))


sim_steps = int(5e4)
pbc = False
time_tag = time.strftime("%Y%m%d-%H%M%S")
amp = 40 #for amp applied on fes. note the gaussian parameters for fes is normalized.

elem = Element(0, "X", "X", 1.0)
top = Topology()
top.addChain()
top.addResidue("xxx", top._chains[0])
top.addAtom("X", elem, top._chains[0]._residues[0])

mass = 12.0 * unit.amu
#starting point as [1.29,-1.29,0.0]
start = Quantity(value = [Vec3(3.14, 3, 0.0)], unit = unit.nanometers) #(value = [Vec3(1.29,-1.29,0.0)], unit = unit.nanometers) #(value = [Vec3(2.12,2.12,0.0)], unit = unit.nanometers)
system = openmm.System()
system.addParticle(mass)

#potential setup
#first we load the gaussians from the file.
# params comes in A, x0, y0, sigma_x, sigma_y format.

gaussian_param = np.loadtxt("./fes_digitize_gauss_params_15_2.txt") 

system = apply_fes(system = system, particle_idx=0, gaussian_param = gaussian_param, pbc = pbc, name = "FES", amp=amp)
z_pot = openmm.CustomExternalForce("100000 * z^2") # very large force constant in z
z_pot.addParticle(0)
system.addForce(z_pot) #on z, large barrier

#pbc section
if pbc:
    a = unit.Quantity((2*np.pi*unit.nanometers, 0*unit.nanometers, 0*unit.nanometers))
    b = unit.Quantity((0*unit.nanometers, 2*np.pi*unit.nanometers, 0*unit.nanometers))
    c = unit.Quantity((0*unit.nanometers, 0*unit.nanometers, 1*unit.nanometers)) # atom not moving in z so we set it to 1 nm
    system.setDefaultPeriodicBoxVectors(a,b,c)


#integrator
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 
                                       1.0/unit.picoseconds, 
                                       0.002*unit.picoseconds)


#before run, last check pbc:
print(system.getDefaultPeriodicBoxVectors())

#CUDA
platform = openmm.Platform.getPlatformByName('CUDA')

#run the simulation
simulation = openmm.app.Simulation(top, system, integrator, platform)
simulation.context.setPositions(start)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
if pbc:
    simulation.context.setPeriodicBoxVectors(a,b,c)


pos_traj = np.zeros([sim_steps, 3])
for i in tqdm(range(sim_steps)):
    simulation.step(1)
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=pbc)
    pos_traj[i,:] = state.getPositions(asNumpy=True)[0,:]

### VISUALIZATION ###

#plot the trajectory over -pi, pi range.
# also plot the fes. loaded in img.
# use scatter plot to plot the traj
gaussian_param = np.loadtxt("./fes_digitize_gauss_params_15_2.txt")
gaussian_param[2::5] = 2*np.pi - gaussian_param[2::5]
gaussian_param[0::5] = gaussian_param[0::5] * amp
n_gaussians = gaussian_param.shape[0]//5
x,y = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, 2*np.pi, 100))
reconstructed = sum_of_gaussians(gaussian_param, x, y, n_gaussians, N=x.shape[0])


plt.figure()
plt.imshow(reconstructed, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp, origin = "lower")
plt.scatter(pos_traj[::10,0], pos_traj[::10,1], s=0.5, alpha=0.5, c='yellow')
plt.xlabel("x")
plt.xlim([-1, 2*np.pi+1])
plt.ylim([-1, 2*np.pi+1])
plt.ylabel("y")
plt.title(f"Unbiased Trajectory, pbc={pbc}")
#plt.show()
plt.savefig(f"./langevin_approach/figs/traj_{time_tag}.png")
plt.close()



#save the plot and the trajectory
np.savetxt(f"./langevin_approach/traj/traj_{time_tag}.txt", pos_traj)

print("all done")

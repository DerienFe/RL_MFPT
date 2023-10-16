#this is a langevin simulator in OPENMM.
# we put a particle in a box and simulate it with Langevin dynamics.
# the external force is defined using a function digitizing the phi/psi fes of dialanine.

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from tqdm import tqdm

import openmm
from openmm import unit
from openmm import Vec3
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.app.metadynamics import BiasVariable, Metadynamics
from openmm.unit import Quantity


def apply_fes(system, particle_idx, gaussian_param, pbc = False, name = "FES", amp = 7):
    """
    this function apply the bias given by the gaussian_param to the system.
    """
    #unpack gaussian parameters
    num_gaussians = int(len(gaussian_param)/5)
    A = gaussian_param[:num_gaussians] * amp #*7
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    y0 = gaussian_param[2*num_gaussians:3*num_gaussians]
    sigma_x = gaussian_param[3*num_gaussians:4*num_gaussians]
    sigma_y = gaussian_param[4*num_gaussians:5*num_gaussians]    
    
    #now we add the force for all gaussians.
    
    for i in range(num_gaussians):
        energy = "0"
        force = openmm.CustomExternalForce(energy)
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

##############################################################
#### INITIALIZATION ####

sim_steps = int(1e5) # 1e7 time_step, 1 step is 2 fs, so 20 ns.
dcd_freq = 1
pbc = True
time_tag = time.strftime("%Y%m%d-%H%M%S")
amp = 1 #for amp applied on fes. note the gaussian parameters for fes is normalized.

#metaD parameters
npoints = 101
meta_freq = 1000
meta_height = 1

#
elem = Element(0, "X", "X", 1.0)
top = Topology()
top.addChain()
top.addResidue("xxx", top._chains[0])
top.addAtom("X", elem, top._chains[0]._residues[0])

mass = 12.0 * unit.amu
#starting point as [1.29,-1.29,0.0]
start = Quantity(value = [Vec3(0.1,2*np.pi-1.29,0.0)], unit = unit.nanometers) #(value = [Vec3(1.29,-1.29,0.0)], unit = unit.nanometers) #(value = [Vec3(2.12,2.12,0.0)], unit = unit.nanometers)
system = openmm.System()
system.addParticle(mass)

##############################################################
####  potential setup ####
#first we load the gaussians from the file.
# params comes in A, x0, y0, sigma_x, sigma_y format.

gaussian_param = np.loadtxt("./fes_digitize_gauss_params_15.txt") 
n_gaussians = int(len(gaussian_param)/5)

system = apply_fes(system = system, particle_idx=0, gaussian_param = gaussian_param, pbc = pbc, name = "FES")
z_pot = openmm.CustomExternalForce("100000 * z^2") # very large force constant in z
z_pot.addParticle(0)
system.addForce(z_pot) #on z, large barrier
#pbc section
if pbc:
    a = unit.Quantity((2*np.pi*unit.nanometers, 0*unit.nanometers, 0*unit.nanometers))
    b = unit.Quantity((0*unit.nanometers, 2*np.pi*unit.nanometers, 0*unit.nanometers))
    c = unit.Quantity((0*unit.nanometers, 0*unit.nanometers, 1*unit.nanometers)) # atom not moving in z so we set it to 1 nm
    system.setDefaultPeriodicBoxVectors(a,b,c)

##############################################################
#### SIMULATION ####

#we initialize the Metadynamics object
#create a folder to store the aux files.
aux_file_path = os.path.join("./langevin_approach/MetaD_auxfiles/" + time_tag)

if not os.path.exists(aux_file_path):
    os.makedirs(aux_file_path)

#apply the rmsd restraint as BiasVariable
target_pos = Quantity(value = [Vec3(5,1,0)], unit = unit.nanometers)

"""
rmsd_cv = openmm.RMSDForce(target_pos, [0])
rmsd_cv.setReferencePositions(target_pos)
print(rmsd_cv.getParticles())
print(rmsd_cv.getReferencePositions())
custom_force = openmm.CustomCVForce("k*RMSD")
custom_force.addGlobalParameter("k", 1.0)
custom_force.addCollectiveVariable("RMSD", rmsd_cv)

rmsd_Bias_var = BiasVariable(custom_force, 1.0, 10, 1, False)
"""
x0, y0 = target_pos[0][0], target_pos[0][1]
if pbc:
    x_cv = openmm.CustomExternalForce("1.0*(periodicdistance(x,0,0, x0,0,0))^2")
else:
    x_cv = openmm.CustomExternalForce("1.0*(x-x0)^2")
x_cv.addGlobalParameter("x0", x0)
x_cv.addParticle(0)
x_force = openmm.CustomCVForce("k*x_cv")
x_force.addGlobalParameter("k", 1.0)
x_force.addCollectiveVariable("x_cv", x_cv)

if pbc:
    y_cv = openmm.CustomExternalForce("1.0*(periodicdistance(0,y,0, 0,y0,0))^2")
else:
    y_cv = openmm.CustomExternalForce("1.0*(y-y0)^2")
y_cv.addGlobalParameter("y0", y0)
y_cv.addParticle(0)
y_force = openmm.CustomCVForce("k*y_cv")
y_force.addGlobalParameter("k", 1.0)
y_force.addCollectiveVariable("y_cv", y_cv)

x_Bias_var = BiasVariable(x_force, 0.0, 25, 1, False) #5^2 = 25
y_Bias_var = BiasVariable(y_force, 0.0, 25, 1, False)

#check forces #the forces are correct. it is possible the BiasVariable is not compatible with ExternalForce class.
#for force in system.getForces():
#    print(force)

metaD = Metadynamics(system=system,
                        variables=[x_Bias_var, y_Bias_var], #variables=[rmsd_Bias_var],
                        temperature=300*unit.kelvin,
                        biasFactor=4,
                        height=meta_height*unit.kilocalorie_per_mole,
                        frequency=meta_freq,
                        saveFrequency=meta_freq,
                        biasDir=aux_file_path,)

platform = openmm.Platform.getPlatformByName('CUDA')

#integrator
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 
                                       1.0/unit.picoseconds, 
                                       0.002*unit.picoseconds)

#run the simulation
simulation = openmm.app.Simulation(top, system, integrator, platform)
simulation.context.setPositions(start)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

pos_traj = np.zeros([sim_steps, 3])
distance = []
#store fes in 2D way
#fes = np.zeros([int(sim_steps/dcd_freq), 50, 50])
potential_energy = []
for i in tqdm(range(int(sim_steps/dcd_freq))):
    metaD.step(simulation, dcd_freq)

    #record the trajectory, distance, and bias applied
    state = simulation.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=pbc)
    pos_traj[i,:] = state.getPositions(asNumpy=True)[0,:]
    distance.append(np.sqrt((pos_traj[i,0] - (-2))**2 + (pos_traj[i,1] - (-2))**2))
    #fes[i,:,:] = metaD.getFreeEnergy()
    energy = state.getPotentialEnergy()
    potential_energy.append(energy)

    #print(f"step {i*dcd_freq}, distance {distance}")

#zip traj, bias, and distance and save.
np.save(f"./langevin_approach/data/traj_{time_tag}.npy", np.array(pos_traj))
np.save(f"./langevin_approach/data/distance_{time_tag}.npy", np.array([distance]))
np.save(f"./langevin_approach/data/potential_energy_{time_tag}.npy", np.array(potential_energy))

#this is for plain MD.
"""
pos_traj = np.zeros([sim_steps, 3])
for i in range(sim_steps):
    simulation.step(1)
    state = simulation.context.getState(getPositions=True)
    pos_traj[i,:] = state.getPositions(asNumpy=True)[0,:]
"""

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

#the image is on -pi to pi, we shift it to 0 to 2pi
img = np.roll(img, int(img.shape[0]/2), axis=0)
img = np.roll(img, int(img.shape[1]/2), axis=1)

#get img square and multiply the amp = 7
min_dim = min(img.shape)
img = img[:min_dim, :min_dim]
img = amp * img
plt.figure()
plt.imshow(img, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp * 12/7)
#plt.scatter(pos_traj[:,0], pos_traj[:,1], s=0.5, alpha=0.5, c='yellow')
#plot every 100 steps
plt.scatter(pos_traj[::5,0], pos_traj[::5,1], s=0.5, alpha=0.5, c='yellow')
#lt.plot(pos_traj[:,0], pos_traj[:,1], c='yellow', alpha = 0.5)
plt.xlabel("x")
plt.xlim([-1, 2*np.pi+1])
plt.ylim([-1, 2*np.pi+1])
plt.ylabel("y")
plt.title("MetaD Trajectory. Target = %s, total step: %s, height: %.2f" % ((target_pos[0][0].value_in_unit(unit.nanometers),target_pos[0][1].value_in_unit(unit.nanometers)), sim_steps, meta_height))
#plt.show()
plt.savefig(f"./langevin_approach/figs/traj_{time_tag}.png")
plt.close()


#visualize fes at first and last frame.
"""plt.figure()
plt.imshow(potential_energy[0], cmap="coolwarm", extent=[-np.pi, np.pi,-np.pi, np.pi], vmin=0, vmax=7)
plt.savefig(f"./langevin_approach/figs/fes_{time_tag}_0.png")
plt.close()

plt.figure()
plt.imshow(potential_energy[-1], cmap="coolwarm", extent=[-np.pi, np.pi,-np.pi, np.pi], vmin=0, vmax=7)
plt.savefig(f"./langevin_approach/figs/fes_{time_tag}_last.png")
plt.close()
"""

print("all done")

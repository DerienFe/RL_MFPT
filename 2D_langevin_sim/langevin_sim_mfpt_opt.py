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

import mdtraj

import config

def apply_fes(system, particle_idx, gaussian_param=None, pbc = False, name = "FES", amp = 7, mode = "gaussian", plot = False, plot_path = "./fes_visualization.png"):
    """
    this function apply the bias given by the gaussian_param to the system.
    """
    #unpack gaussian parameters
    if mode == "gaussian":
        num_gaussians = int(len(gaussian_param)/5)
        A = gaussian_param[0::5] * amp #*7
        x0 = gaussian_param[1::5]
        y0 = gaussian_param[2::5]
        sigma_x = gaussian_param[3::5]
        sigma_y = gaussian_param[4::5]

        #now we add the force for all gaussians.
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
        if plot:
            #plot the fes.
            x = np.linspace(0, 2*np.pi, 100)
            y = np.linspace(0, 2*np.pi, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i in range(num_gaussians):
                Z += A[i] * np.exp(-(X-x0[i])**2/(2*sigma_x[i]**2) - (Y-y0[i])**2/(2*sigma_y[i]**2))
            plt.figure()
            plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
            plt.xlabel("x")
            plt.xlim([-1, 2*np.pi+1])
            plt.ylim([-1, 2*np.pi+1])
            plt.ylabel("y")
            plt.title("FES mode = gaussian, pbc=False")
            plt.colorbar()
            plt.savefig(plot_path)
            plt.close()

    if mode == "multiwell":
        """
        here we create a multiple well potential.
         essentially we deduct multiple gaussians from a flat surface, 
         with a positive gaussian acting as an additional barrier.
         note we have to implement this into openmm CustomExternalForce.
            the x,y is [0, 2pi]
         eq:
            U(x,y) = amp * (1                                                                   #flat surface
                            - A_i*exp(-(x-x0i)^2/(2*sigma_xi^2) - (y-y0i)^2/(2*sigma_yi^2))) ...        #deduct gaussians
                            + A_j * exp(-(x-x0j)^2/(2*sigma_xj^2) - (y-y0j)^2/(2*sigma_yj^2))       #add a sharp positive gaussian
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for multi-well potential.")
        else:
            num_wells = 9
            num_barrier = 1

            #here's the well params
            A_i = np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp * 4.184
            x0_i = [1, 1, 3, 4, 4, 5, 5.5, 6, 1]
            y0_i = [1, 2, 2, 3, 5, 4, 5.5, 1, 5]
            sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]
            sigma_y_i = [0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7]

            #here's the barrier params
            # for example we define a diagonal barrier at x = pi
            A_j = np.array([0.3]) * amp * 4.184
            x0_j = [np.pi]
            y0_j = [np.pi]
            sigma_x_j = [3]
            sigma_y_j = [0.3]

            #now we add the force for all gaussians.
            #note all energy is in Kj/mol unit.
            energy = str(amp * 4.184) #flat surface
            force = openmm.CustomExternalForce(energy)
            for i in range(num_wells):
                energy = f"-A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2) - (y-y0{i})^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)

                #examine the current energy term within force.

                print(force.getEnergyFunction())

                force.addGlobalParameter(f"A{i}", A_i[i])
                force.addGlobalParameter(f"x0{i}", x0_i[i])
                force.addGlobalParameter(f"y0{i}", y0_i[i])
                force.addGlobalParameter(f"sigma_x{i}", sigma_x_i[i])
                force.addGlobalParameter(f"sigma_y{i}", sigma_y_i[i])
                force.addParticle(particle_idx)
                #we append the force to the system.
                system.addForce(force)
            
            for i in range(num_barrier):
                energy = f"A{i+num_wells}*exp(-(x-x0{i+num_wells})^2/(2*sigma_x{i+num_wells}^2) - (y-y0{i+num_wells})^2/(2*sigma_y{i+num_wells}^2))"
                force = openmm.CustomExternalForce(energy)

                #examine the current energy term within force.

                print(force.getEnergyFunction())

                force.addGlobalParameter(f"A{i+num_wells}", A_j[i])
                force.addGlobalParameter(f"x0{i+num_wells}", x0_j[i])
                force.addGlobalParameter(f"y0{i+num_wells}", y0_j[i])
                force.addGlobalParameter(f"sigma_x{i+num_wells}", sigma_x_j[i])
                force.addGlobalParameter(f"sigma_y{i+num_wells}", sigma_y_j[i])
                force.addParticle(particle_idx)
                #we append the force to the system.
                system.addForce(force)
            
            if plot:
                #plot the fes.
                x = np.linspace(0, 2*np.pi, 100)
                y = np.linspace(0, 2*np.pi, 100)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                Z += amp * 4.184 #flat surface
                for i in range(num_wells):
                    Z -= A_i[i] * np.exp(-(X-x0_i[i])**2/(2*sigma_x_i[i]**2) - (Y-y0_i[i])**2/(2*sigma_y_i[i]**2))
                for i in range(num_barrier):
                    Z += A_j[i] * np.exp(-(X-x0_j[i])**2/(2*sigma_x_j[i]**2) - (Y-y0_j[i])**2/(2*sigma_y_j[i]**2))
                plt.figure()
                plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7 * 4.184, origin="lower")
                plt.xlabel("x")
                plt.xlim([-1, 2*np.pi+1])
                plt.ylim([-1, 2*np.pi+1])
                plt.ylabel("y")
                plt.title("FES mode = multiwell, pbc=False")
                plt.colorbar()
                plt.savefig(plot_path)
                plt.close()
            
    if mode == "funnel":
        """
        this is funnel like potential.
        we start wtih a flat fes, then add/deduct sphrical gaussians
        eq:
            U = 0.7* amp * cos(2 * p * (sqrt((x-pi)^2 + (y-pi)^2))) #cos function. periodicity determines the num of waves.
            - amp exp(-((x-pi)^2+(y-pi)^2))
            + 0.4*amp*((x-pi/8)^2 + (y-pi/8)^2)
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for funnel potential.")
        else:
            plot_3d = True
            periodicity = 8
            energy = f"0.7*{amp} * cos({periodicity} * (sqrt((x-pi)^2 + (y-pi)^2))) - 0.6* {amp} * exp(-((x-pi)^2+(y-pi)^2)) + 0.4*{amp}*((x-pi/8)^2 + (y-pi/8)^2)"
            force = openmm.CustomExternalForce(energy)
            force.addParticle(particle_idx)
            system.addForce(force)
            if plot:
                if plot_3d:
                    import plotly.graph_objs as go

                    # Define the x, y, and z coordinates
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.7* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z -= 0.6* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2)/0.5)
                    Z += 0.4*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    # Create the 3D contour plot
                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, cmin = 0, cmax = amp *12/7)])
                    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
                    fig.update_layout(title='FES mode = funnel, pbc=False', autosize=True,
                                    width=800, height=800,
                                    scene = {
                                        "xaxis": {"nticks": 5},
                                        "yaxis": {"nticks": 5},
                                        "zaxis": {"nticks": 5},
                                        "camera_eye": {"x": 1, "y": 1, "z": 0.4},
                                        "aspectratio": {"x": 1, "y": 1, "z": 0.4}
                                    }
                                    )
                                    #margin=dict(l=65, r=50, b=65, t=90))
                    #save fig.
                    fig.write_image(plot_path)
                    
                else:
                    #plot the fes.
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.4* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z += 0.7* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2/0.5))
                    Z += 0.2*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    plt.figure()
                    plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
                    plt.xlabel("x")
                    plt.xlim([-1, 2*np.pi+1])
                    plt.ylim([-1, 2*np.pi+1])
                    plt.ylabel("y")
                    plt.title("FES mode = funnel, pbc=False")
                    plt.colorbar()
                    plt.savefig(plot_path)
                    plt.close()

    #at last we add huge barrier at the edge of the box. since we are not using pbc.
    #this is to prevent the particle from escaping the box.
    left_pot = openmm.CustomExternalForce("100000 * step(-x)")
    right_pot = openmm.CustomExternalForce("100000 * step(x - 2*pi)")
    bottom_pot = openmm.CustomExternalForce("100000 * step(-y)")
    top_pot = openmm.CustomExternalForce("100000 * step(y - 2*pi)")

    left_pot.addParticle(particle_idx)
    right_pot.addParticle(particle_idx)
    bottom_pot.addParticle(particle_idx)
    top_pot.addParticle(particle_idx)

    system.addForce(left_pot)
    system.addForce(right_pot)
    system.addForce(bottom_pot)
    system.addForce(top_pot)

    return system

def apply_bias(system, particle_idx, gaussian_param, pbc = False, name = "BIAS", amp = 1, num_gaussians = 20):
    
    """
    this applies a bias using customexternal force class. similar as apply_fes.
    note this leaves a set of global parameters Ag, x0g, y0g, sigma_xg, sigma_yg.
    as these parameters can be called and updated later.
    note this is done while preparing the system before assembling the context.
    """
    assert len(gaussian_param) == 5 * num_gaussians, "gaussian_param should be in A, x0, y0, sigma_x, sigma_y format."

    #unpack gaussian parameters
    A = gaussian_param[0::num_gaussians] * amp #*7
    x0 = gaussian_param[1::num_gaussians]
    y0 = gaussian_param[2::num_gaussians]
    sigma_x = gaussian_param[3::num_gaussians]
    sigma_y = gaussian_param[4::num_gaussians]

    #now we add the force for all gaussians. with num_gaussians terms.
    energy = "0"
    force = openmm.CustomExternalForce(energy)
    for i in range(num_gaussians):
        if pbc:
            energy = f"Ag{i}*exp(-periodicdistance(x,0,0, x0g{i},0,0)^2/(2*sigma_xg{i}^2) - periodicdistance(0,y,0, 0,y0g{i},0)^2/(2*sigma_yg{i}^2))"
            force = openmm.CustomExternalForce(energy)
        else:
            energy = f"Ag{i}*exp(-(x-x0g{i})^2/(2*sigma_xg{i}^2) - (y-y0g{i})^2/(2*sigma_yg{i}^2))"
            force = openmm.CustomExternalForce(energy)

        #examine the current energy term within force.

        print(force.getEnergyFunction())

        force.addGlobalParameter(f"Ag{i}", A[i])
        force.addGlobalParameter(f"x0g{i}", x0[i])
        force.addGlobalParameter(f"y0g{i}", y0[i])
        force.addGlobalParameter(f"sigma_xg{i}", sigma_x[i])
        force.addGlobalParameter(f"sigma_yg{i}", sigma_y[i])
        force.addParticle(particle_idx)
        #we append the force to the system.
        system.addForce(force)
    return system

def update_bias(simulation, gaussian_param, name = "BIAS", amp = 1, num_gaussians = 20):
    """
    given the gaussian_param, update the bias.
    note this requires the context object. or a simulation object.
    # the context object can be accessed by simulation.context.
    """
    assert len(gaussian_param) == 5 * num_gaussians, "gaussian_param should be in A, x0, y0, sigma_x, sigma_y format."

    #unpack gaussian parameters
    A = gaussian_param[0::num_gaussians] * amp #*7
    x0 = gaussian_param[1::num_gaussians]
    y0 = gaussian_param[2::num_gaussians]
    sigma_x = gaussian_param[3::num_gaussians]
    sigma_y = gaussian_param[4::num_gaussians]

    #now we update the GlobalParameter for all gaussians. with num_gaussians terms. and update them in the system.
    for i in range(num_gaussians):
        simulation.context.setParameter(f"Ag{i}", A[i])
        simulation.ontext.setParameter(f"x0g{i}", x0[i])
        simulation.context.setParameter(f"y0g{i}", y0[i])
        simulation.context.setParameter(f"sigma_xg{i}", sigma_x[i])
        simulation.context.setParameter(f"sigma_yg{i}", sigma_y[i])
    
    return simulation

def propagate(simulation,
              gaussian_params,
              prop_index, 
              pos_traj,   #this records the trajectory of the particle. in shape: [prop_index, sim_steps, 3]
              steps=config.propagation_step,
              dcdfreq=config.dcdfreq,
              platform=config.platform,
              stepsize=config.stepsize,
              num_bins=config.num_bins,
              pbc=config.pbc,
              ):
    """
    here we use the openmm context object to propagate the system.
    save the CV and append it into the CV_total.
    use the DHAM_it to process CV_total, get the partially observed Markov matrix from trajectory.
    return the current position, the CV_total, and the partially observed Markov matrix.
    """
    
    file_handle = open(f"./trajectory/langevin_sim_mfpt_opt_{prop_index}.dcd", 'bw')
    dcd_file = openmm.app.dcdfile.DCDFile(file_handle, top, dt = stepsize) #note top is global param, as we constructed the system.

    for _ in tqdm(range(int(steps/dcdfreq)), desc=f"Propagation {prop_index}"):
        simulation.integrator.step(dcdfreq)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=pbc)
        dcd_file.writeModel(state.getPositions(asNumpy=True))
    file_handle.close()

    #use mdtraj to get the coordinate of the particle.
    traj = mdtraj.load_dcd(f"./trajectory/langevin_sim_mfpt_opt_{prop_index}.dcd", top=top)
    coor = traj.xyz[:,0,:] #[all_frames,particle_index,xyz] # we grep the particle 0.
    #we digitize the x, y coordinate into meshgrid (0, 2pi, num_bins)
    x = np.linspace(0, 2*np.pi, num_bins)
    y = np.linspace(0, 2*np.pi, num_bins)
    #we digitize the coor into the meshgrid.
    coor_xy = coor.squeeze()[:,:2] #we only take the x, y coordinate.
    coor_x_digitized = np.digitize(coor_xy[:,0], x)
    coor_y_digitized = np.digitize(coor_xy[:,1], y)
    coor_xy_digitized = np.stack([coor_x_digitized, coor_y_digitized], axis=1) #shape: [all_frames, 2]

    #check!
    print(coor_xy_digitized.shape)

    #we append the coor_xy_digitized into the pos_traj.
    pos_traj[prop_index,:,:] = coor_xy_digitized

    #combine the pos_traj until this prop_index and feed it into the DHAM.
    coor_xy_digitized_total = pos_traj[:prop_index+1,:,:].reshape(-1,2) #we reshape it into [prop_index * sim_steps, 2]

    #we ravel the coor_xy_digitized_total into 1D using np.ravel_multi_index, order = C.
    coor_xy_digital_ravelled_total = np.ravel_multi_index(coor_xy_digitized_total, (num_bins, num_bins), order='C') #shape [prop_index * sim_steps, 1]

    #here we use the DHAM.
    F_M, MM = DHAM_it(coor_xy_digital_ravelled_total.reshape(-1,1), gaussian_params, T=300, lagtime=1, numbins=num_bins)
    
    return coor_xy_digital_ravelled_total[-1], MM, pos_traj #return the last position, the MM, CV traj list.

elem = Element(0, "X", "X", 1.0)
top = Topology()
top.addChain()
top.addResidue("xxx", top._chains[0])
top.addAtom("X", elem, top._chains[0]._residues[0])
mass = 12.0 * unit.amu
system = openmm.System()
system.addParticle(mass)
gaussian_param = np.loadtxt("./params/gaussian_fes_param.txt")
system = apply_fes(system = system, particle_idx=0, gaussian_param = gaussian_param, pbc = config.pbc, name = "FES", mode="funnel", plot = True)
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

#create simulation object, this create a context object automatically.
# when we need to pass a context object, we can pass simulation instead.
simulation = openmm.app.Simulation(top, system, integrator, config.platform)
simulation.context.setPositions(config.start)
simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
if config.pbc:
    simulation.context.setPeriodicBoxVectors(a,b,c)

pos_traj = np.zeros([config.sim_steps/config.propagation_step, config.propagation_step, 3]) #initialize the pos_traj with zero.




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
img = config.amp * img
plt.figure()
plt.imshow(img, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=config.amp *12/7)
plt.scatter(pos_traj[:,0], pos_traj[:,1], s=0.5, alpha=0.5, c='yellow')
plt.xlabel("x")
plt.xlim([-1, 2*np.pi+1])
plt.ylim([-1, 2*np.pi+1])
plt.ylabel("y")
plt.title("Unbiased Trajectory, pbc=True")
#plt.show()
plt.savefig(f"./langevin_approach/figs/traj_{config.time_tag}.png")
plt.close()



#save the plot and the trajectory
np.savetxt(f"./langevin_approach/traj/traj_{config.time_tag}.txt", pos_traj)

print("all done")

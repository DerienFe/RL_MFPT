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
import csv

import config
from dham import *
from util import *

def propagate(simulation,
              prop_index,
              cycle_count,
              pos_traj,   #this records the trajectory of the particle. in shape: [prop_index, sim_steps, 3]
              steps=config.propagation_step,
              dcdfreq=config.dcdfreq_mfpt,
              stepsize=config.stepsize,
              num_bins=config.num_bins,
              pbc=config.pbc,
              time_tag=None,
              top=None,
              reach=None,
              current_target_state = None,
              ):
    """
    here we use the openmm context object to propagate the system.
    save the CV and append it into the CV_total.
    use the DHAM_it to process CV_total, get the partially observed Markov matrix from trajectory.
    return the current position, the CV_total, and the partially observed Markov matrix.
    """
    print("propagating")
    print("current target state: ", current_target_state)
    print("cycle count: ", cycle_count)
    print("prop index: ", prop_index)
    file_handle = open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_cyc_{cycle_count}_prop_{i_prop}.dcd", 'bw')
    dcd_file = openmm.app.dcdfile.DCDFile(file_handle, top, dt = stepsize) #note top is no longer a global pararm, we need pass this.
    for _ in tqdm(range(int(steps/dcdfreq)), desc=f"cycle {cycle_count} Propagation {prop_index}"):
        simulation.integrator.step(dcdfreq)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=pbc)
        dcd_file.writeModel(state.getPositions(asNumpy=True))
    file_handle.close()

    #save the top to pdb.
    with open(f"./trajectory/explore/{time_tag}_langevin_sim_explore_cyc_{cycle_count}_prop_{i_prop}.pdb", 'w') as f:
        openmm.app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    
    #we load the pdb and pass it to mdtraj_top
    mdtraj_top = mdtraj.load(f"./trajectory/explore/{time_tag}_langevin_sim_explore_cyc_{cycle_count}_prop_{i_prop}.pdb")

    #use mdtraj to get the coordinate of the particle.
    traj = mdtraj.load_dcd(f"./trajectory/explore/{time_tag}_langevin_sim_explore_cyc_{cycle_count}_prop_{i_prop}.dcd", top = mdtraj_top)#top = mdtraj.Topology.from_openmm(top)) #this will yield error because we using imaginary element X.
    coor = traj.xyz[:,0,:] #[all_frames,particle_index,xyz] # we grep the particle 0.

    #we digitize the x, y coordinate into meshgrid (0, 2pi, num_bins)


    #we digitize the coor into the meshgrid.
    coor_x = coor.squeeze()[:,:1] #we only take the xcoordinate.
    #we test.
    if False: 
        plt.figure()
        plt.plot(coor_x)
        plt.xlim([0, 2*np.pi])
        plt.savefig("./test.png")
        plt.close()
    #we append the coor_xy_digitized into the pos_traj.
    pos_traj[prop_index,:] = coor_x.squeeze()

    #we take all previous digitized x and feed it into DHAM.
    coor_x_total = pos_traj[:prop_index+1,:].squeeze() #note this is in coordinate space np.linspace(0, 2*np.pi, num_bins)
    coor_x_total = coor_x_total.reshape(prop_index+1, -1, 1)
    print("coor_x_total shape feed into dham: ", coor_x_total.shape)

    #here we load all gaussian_params.
    gaussian_params = np.zeros([prop_index+1, config.num_gaussian, 3])
    for i in range(prop_index+1):
        gaussian_params[i,:,:] = np.loadtxt(f"./params/{time_tag}_gaussian_param_prop_{i}.txt").reshape(-1,3)
        print(f"gaussian_params for propagation {i} loaded.")

    #here we use the DHAM.
    F_M, MM = DHAM_it(coor_x_total, 
                      gaussian_params, 
                      T=300, 
                      lagtime=1, 
                      num_bins=num_bins, 
                      time_tag=time_tag, 
                      prop_index=prop_index,
                      use_symmetry=True
                      )
    cur_pos = coor_x_total[-1] #the current position of the particle, in ravelled 1D form.
    
    #determine if the particle has reached the target state.
    end_state_xyz = current_target_state.value_in_unit_system(openmm.unit.md_unit_system)[0] #config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0]
    end_state_x = end_state_xyz[:1]
    for index_d, d in enumerate(coor_x):
        #if the distance of current pos is the config.target_state, we set reach to index_d.
        target_distance = np.linalg.norm(d - end_state_x)
        if target_distance < 0.1:
            reach = index_d * config.dcdfreq_mfpt

    return cur_pos, pos_traj, MM, reach, F_M

def get_working_MM(M):
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

def find_closest_index(working_indices, final_index, N):
    """
    returns the farest index in 1D.

    here we find the closest state to the final state.
    first we unravel all the index to 2D.
    then we use the lowest RMSD distance to find the closest state.
    then we ravel it back to 1D.
    note: for now we only find the first-encounted closest state.
          we can create a list of all the closest states, and then choose random one.
    """
    def rmsd_dist(a, b):
        return np.sqrt(np.sum((a-b)**2))
    working_x, working_y = np.unravel_index(working_indices, (N,N), order='C')
    working_states = np.stack((working_x, working_y), axis=1)
    final_state = np.unravel_index(final_index, (N,N), order='C')
    closest_state = working_states[0]
    for i in range(len(working_states)):
        if rmsd_dist(working_states[i], final_state) < rmsd_dist(closest_state, final_state):
            closest_state = working_states[i]
        
    closest_index = np.ravel_multi_index(closest_state, (N,N), order='C')
    return closest_index

def DHAM_it(CV, gaussian_params, T=300, lagtime=2, num_bins=150, prop_index=0, time_tag=None, use_symmetry=True):
    """
    intput:
    CV: the collective variable we are interested in. now it's 2d.
    gaussian_params: the parameters of bias potential. (in our case the 10-gaussian params)
     format: (a,bx, by,cx,cy)
    T: temperature 300

    output:
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = DHAM(gaussian_params, num_bins=num_bins)
    d.setup(CV, T, prop_index=prop_index, time_tag=time_tag)

    d.lagtime = lagtime
    d.num_bins = num_bins #num of bins, arbitrary.
    results = d.run(biased = True, plot=True, use_symmetry=use_symmetry)
    return results

def random_initial_bias(initial_position, num_gaussians):
    #returns random a,b,c for 10 gaussian functions. around the initial position
    # initial position is in Anstrom
    initial_position = initial_position.value_in_unit_system(openmm.unit.md_unit_system)[0] #this is in nm
    rng = np.random.default_rng()
    #a = np.ones(10)
    a = np.ones(num_gaussians) * 0.01 
    b = rng.uniform(initial_position[0]-0.5, initial_position[0]+0.5, num_gaussians)
    c = rng.uniform(0, 2*np.pi, num_gaussians)
    return np.concatenate((a,b,c), axis=None)
    

if __name__ == "__main__":
    cycle_count = 0 #for back n forth.
    current_target_state = config.end_state

    elem = Element(0, "X", "X", 1.0)
    top = Topology()
    top.addChain()
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X", elem, top._chains[0]._residues[0])
    mass = 12.0 * unit.amu
    for i_sim in range(config.num_sim):
    #def simulate_once():
        print("system initializing")
        #print out all the config.
        print("config: ", config.__dict__)
        
        time_tag = time.strftime("%Y%m%d-%H%M%S")

        #print current time tag.
        print("time_tag: ", time_tag)

        system = openmm.System() #we initialize the system every
        system.addParticle(mass)
        #gaussian_param = np.loadtxt("./params/gaussian_fes_param.txt")
        system, fes = apply_fes(system = system, 
                                particle_idx=0, 
                                gaussian_param = None, 
                                pbc = config.pbc, 
                                amp = config.amp, 
                                name = "FES",
                                mode=config.fes_mode, 
                                plot = True)
        y_pot = openmm.CustomExternalForce("1e3 * y^2") # very large force constant in y
        y_pot.addParticle(0)
        z_pot = openmm.CustomExternalForce("1e3 * z^2") # very large force constant in z
        z_pot.addParticle(0)
        system.addForce(z_pot) #on z, large barrier
        system.addForce(y_pot) #on y, large barrier

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

        num_propagation = 100 #hard coded for now, we give it 10 propagation per cycle, and 10 cycles.
        frame_per_propagation = int(config.propagation_step/config.dcdfreq_mfpt)
        #this stores the digitized, ravelled, x, y coordinates of the particle, for every propagation.
        pos_traj = np.zeros([num_propagation, frame_per_propagation]) #shape: [num_propagation, frame_per_propagation]

        x = np.linspace(0, 2*np.pi, config.num_bins)

        #we start propagation.
        #note num_propagation = config.sim_steps/config.propagation_step
        reach = None
        i_prop = 0
        #for i_prop in range(num_propagation):
        while cycle_count < config.max_cycle:
            while reach is None:
                if i_prop >= num_propagation:
                    print("propagation number exceeds num_propagation, break")
                    break
                if i_prop == 0:
                    print("propagation 0 starting")
                    gaussian_params = random_initial_bias(initial_position = config.start_state, num_gaussians = config.num_gaussian)
                    np.savetxt(f"./params/{time_tag}_gaussian_param_prop_{i_prop}.txt", gaussian_params)
                    #we apply the initial gaussian bias (v small) to the system
                    system = apply_bias(system = system, particle_idx=0, gaussian_param = gaussian_params, pbc = config.pbc, name = "BIAS", num_gaussians = config.num_gaussian)

                    #create simulation object, this create a context object automatically.
                    # when we need to pass a context object, we can pass simulation instead.
                    simulation = openmm.app.Simulation(top, system, integrator, config.platform)
                    simulation.context.setPositions(config.start_state)
                    simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

                    simulation.minimizeEnergy()
                    if config.pbc:
                        simulation.context.setPeriodicBoxVectors(a,b,c)

                    #now we propagate the system, i.e. run the langevin simulation.
                    cur_pos, pos_traj, MM, reach, F_M = propagate(simulation = simulation,
                                                                        gaussian_params = gaussian_params,
                                                                        prop_index = i_prop,
                                                                        cycle_count = cycle_count,
                                                                        pos_traj = pos_traj,
                                                                        steps=config.propagation_step,
                                                                        dcdfreq=config.dcdfreq_mfpt,
                                                                        stepsize=config.stepsize,
                                                                        num_bins=config.num_bins,
                                                                        pbc=config.pbc,
                                                                        time_tag = time_tag,
                                                                        top=top,
                                                                        reach=reach,
                                                                        current_target_state = current_target_state
                                                                        )

                    working_MM, working_indices = get_working_MM(MM)

                    #final_coor = config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0][:1]
                    #we adapt this to current_target_state
                    final_coor = current_target_state.value_in_unit_system(openmm.unit.md_unit_system)[0][:1]

                    final_index = np.digitize(final_coor, x)
                    closest_index = working_indices[np.argmin(np.abs(working_indices - final_index))] #find the closest index in working_indices to final_index.
                    print("closest_index updated: ", closest_index)
                    print("converted to x space that's: ", x[closest_index])
                    i_prop += 1
                else:

                    print(f"cycle {cycle_count} propagation number {i_prop} starting")

                    #find the most visited state in last propagation.
                    last_traj = pos_traj[i_prop-1, :].squeeze()
                    last_traj_index = np.digitize(last_traj, x).astype(np.int64)
                    most_visited_state = np.argmax(np.bincount(last_traj_index)) #this is in digitized
                    last_visited_state = last_traj_index[-1] #this is in digitized

                    gaussian_params = try_and_optim_M(working_MM,
                                                    working_indices = working_indices,
                                                    num_gaussian = config.num_gaussian,
                                                    start_index = last_visited_state,
                                                    end_index = closest_index,
                                                    plot = False,
                                                    )
                    if True:
                        #here we calculate the total bias given the optimized gaussian_params
                        total_bias = get_total_bias(x, gaussian_params, num_gaussians=config.num_gaussian) * 4.184 #convert to kcal/mol
                        plt.figure()
                        plt.plot(x, total_bias, label="total bias applied")
                        #plt.savefig(f"./figs/explore/{time_tag}_total_bias_{i_prop}.png")
                        #plt.close()
                        
                        #here we plot the reconstructed fes from MM.
                        # we also plot the most_visited_state and closest_index.
                        #plt.figure()
                        plt.plot(x, F_M*4.184, label="DHAM fes")
                        #plt.plot(x[most_visited_state], F_M[most_visited_state]*4.184, marker='o', markersize=3, color="blue", label = "most visited state (local start)")
                        plt.plot(x[last_visited_state], F_M[last_visited_state]*4.184, marker='o', markersize=3, color="blue", label = "last visited state (local start)")
                        plt.plot(x[closest_index], F_M[closest_index]*4.184, marker='o', markersize=3, color="red", label = "closest state (local target)")
                        #plt.legend()
                        #plt.savefig(f"./figs/explore/{time_tag}_reconstructed_fes_{i_prop}.png")
                        #plt.close()

                        #we plot here to check the original fes, total_bias and trajectory.
                    
                        #we add the total bias to the fes.
                        #fes += total_bias_big
                        #plt.figure()
                        plt.plot(x, fes, label="original fes")
                        plt.xlabel("x-coor position (nm)")
                        #plt.xlim([-1, 2*np.pi+1])
                        #plt.ylim([-1, 2*np.pi+1])
                        plt.ylabel("fes (kJ/mol)")
                        plt.title("FES mode = multiwell, pbc=False")
                        
                        #additionally we plot the trajectory.
                        # first we process the pos_traj into x, y coordinate.
                        # we plot all, but highlight the last prop_step points with higher alpha.

                        #history_traj = pos_traj[:i_prop, :].squeeze() #note this is only the x coor.
                        #recent_traj = pos_traj[i_prop:, :].squeeze()

                        history_traj = pos_traj[:i_prop, :].squeeze() #note this is only the x coor.
                        recent_traj = pos_traj[i_prop:, :].squeeze()

                        #here we digitize this so we can plot it to fes.
                        history_traj = np.digitize(history_traj, x)
                        recent_traj = np.digitize(recent_traj, x)

                        plt.scatter(x[history_traj], fes[history_traj], s=3.5, alpha=0.3, c='grey')
                        plt.scatter(x[recent_traj], fes[recent_traj], s=3.5, alpha=0.8, c='black')
                        plt.legend(loc='upper left')
                        plt.savefig(f"./figs/explore/{time_tag}_fes_traj_cyc_{cycle_count}_prop_{i_prop}.png")
                        plt.close()

                    #save the gaussian_params
                    np.savetxt(f"./params/{time_tag}_gaussian_param_prop_{i_prop}.txt", gaussian_params)

                    #apply the gaussian_params to openmm system.
                    simulation = update_bias(simulation = simulation,
                                            gaussian_param = gaussian_params,
                                            name = "BIAS",
                                            num_gaussians=config.num_gaussian,
                                            )
                    
                    #we propagate system again
                    cur_pos, pos_traj, MM, reach, F_M = propagate(simulation = simulation,
                                                                        gaussian_params = gaussian_params,
                                                                        prop_index = i_prop,
                                                                        cycle_count = cycle_count,
                                                                        pos_traj = pos_traj,
                                                                        steps=config.propagation_step,
                                                                        dcdfreq=config.dcdfreq_mfpt,
                                                                        stepsize=config.stepsize,
                                                                        num_bins=config.num_bins,
                                                                        pbc=config.pbc,
                                                                        time_tag = time_tag,
                                                                        top=top,
                                                                        reach=reach,
                                                                        current_target_state = current_target_state
                                                                        )
                    #update working_MM and working_indices
                    working_MM, working_indices = get_working_MM(MM)
                    #update closest_index
                    final_coor = current_target_state.value_in_unit_system(openmm.unit.md_unit_system)[0][:1]
                    final_index = np.digitize(final_coor, x)
                    closest_index = working_indices[np.argmin(np.abs(working_indices - final_index))] #find the closest index in working_indices to final_index.
                    print("closest_index updated: ", closest_index)
                    print("converted to x space that's: ", x[closest_index])
                    i_prop += 1
            
            #here we reached the current_target_state.
            # we now reverse the direction. set current_target_state to start_state or end_state.
            if np.allclose(current_target_state.value_in_unit_system(openmm.unit.md_unit_system)[0], config.end_state.value_in_unit_system(openmm.unit.md_unit_system)[0]):
                current_target_state = config.start_state
            else:
                current_target_state = config.end_state
            cycle_count += 1
            reach = None
            #i_prop = 0 #we keep the i_prop, so its easier to reshape the pos_traj.
            print("cycle number completed: ", cycle_count)

            #we also updadte the closest_index.
            #update working_MM and working_indices
            working_MM, working_indices = get_working_MM(MM)
            #update closest_index
            final_coor = current_target_state.value_in_unit_system(openmm.unit.md_unit_system)[0][:1]
            final_index = np.digitize(final_coor, x)
            closest_index = working_indices[np.argmin(np.abs(working_indices - final_index))] #find the closest index in working_indices to final_index.    
            print("closest_index updated: ", closest_index)
            print("converted to x space that's: ", x[closest_index])
            
            #each time we finish a cycle, we plot the total F_M.
            #here we load all gaussian_params.
            gaussian_params = np.zeros([i_prop, config.num_gaussian, 3])
            for i in range(i_prop):
                gaussian_params[i,:,:] = np.loadtxt(f"./params/{time_tag}_gaussian_param_prop_{i}.txt").reshape(-1,3)
                print(f"gaussian_params for propagation {i} loaded.")

            #use DHAM_it to get the MM and F_M.
            F_M_plot, _ = DHAM_it(pos_traj[:i_prop,:].ravel().reshape(i_prop, -1, 1), 
                                gaussian_params, 
                                T=300, 
                                lagtime=1, 
                                num_bins=config.num_bins, 
                                time_tag=time_tag, 
                                prop_index=0,
                                use_symmetry=False
                                )
            F_M_cleaned = np.where(np.isfinite(F_M_plot), F_M_plot, np.nan)
            #plot the total F_M at last.
            plt.figure()
            plt.plot(x, (F_M_plot - np.nanmin(F_M_cleaned)), label="DHAM fes")
            plt.plot(x,fes - fes.min(), label="original fes")
            plt.title("total F_M")
            plt.xlabel("x-coor position (nm)")
            plt.ylabel("fes (kcal/mol)")
            plt.legend()
            plt.savefig(f"./figs/explore/{time_tag}_cycle_{cycle_count}_total_fes.png")
            plt.close()

        ############# post processing #################
        #we have reached target state, thus we record the steps used.
        total_steps = i_prop * config.propagation_step
        print("total steps used: ", total_steps)

        with open("./total_steps_mfpt.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([total_steps])
        np.savetxt(f"./visited_states/{time_tag}_pos_traj.txt", np.ravel(pos_traj))
print("all done")
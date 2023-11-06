#Here we explore the NaCl FES energy landscape on Na-Cl distance. place random bias.
# the workflow:
# propagate the system for a number of steps.
#    we have to start from loading the molecule, FF
#    then we put bias, log the bias form in txt file. (random guess around starting position if first propagation)
#    then we propagate the system for a number of steps.
#    use DHAM, feed in the NaCl distance, bias params for current propagation, get free energy landscape
#    use the partial free energy landscape to generate next bias params.
#    repeat.
## import required packages

import time
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import openmm.app as omm_app
import openmm as omm
import openmm.unit as unit
import mdtraj

from util import *
from openmm.app import *
from dham import DHAM
from tqdm import tqdm
from matplotlib import animation

sys.path.append("..")
platform = omm.Platform.getPlatformByName('CUDA')
psf_file = 'step3_input.psf' # Path
pdb_file = 'traj_0.restart.pdb' #'step3_input.pdb' # Path
T = 298.15      # temperature in K
fricCoef = 10   # friction coefficient in 1/ps
stepsize = 2    # MD integration step size in fs
dcdfreq = 100    #100   # save coordinates at every 100 step
minimizing = False #we don't minimize, we use the traj_restart.pdb.
propagation_step = 5000
max_propagation = 100
num_simulations = 5
num_bins = 100 #for qspace used in DHAM and etc.
num_frames = []
fig, ax = plt.subplots()

env = os.environ.copy()

matrices = []
matricies_analytic =[]

frames = []

# Set up the plot
fig, ax = plt.subplots()

"""
if len(sys.argv) != 3:
    print("Expected 2 arguments: run_type, run_number")
    sys.exit(1)

run_type = sys.argv[1]
run_number = int(sys.argv[2])


print(f"Run Type: {run_type}, Run Number: {run_number}")
"""
def animate(i):
    plt.imshow(matrices[i], cmap='hot')


def last_dcd_frames(fin_index): 
    print("In lasts dcd frames")
    top = mdtraj.load_psf(psf_file)
    print("Loaded psf file")
    traj = mdtraj.load_dcd(f"trajectories/explore_traj/NaCl_exploring_traj_{fin_index-1}.dcd", top=top)
    print("Loaded dcd file")
    dists = mdtraj.compute_distances(traj, [[0, 1]]) *10 #unit in A #get distance over the traj (in this propagation)
    print("this is fin  index: ", fin_index, "theses are the dists: ", dists)

    for frame, dist in enumerate(dists):
        if dist >= 7:
            print("this is frames:", frame)
            return frame 


def update_metrics(run_type, run_number, frames, time):
    filename = 'mm_metrics.csv'
    file_exists = os.path.isfile(filename)

    # Open the CSV file in append mode ('a'), so new rows can be added at the end
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file didn't exist, write the header
        if not file_exists:
            writer.writerow(['Run Type', 'Run Number', 'NumFrames', 'Time'])

        # Write the new row with the provided data
        writer.writerow([run_type, run_number, frames,time])


def random_initial_bias(initial_position):
    #returns random a,b,c for 10 gaussian functions. around the initial position
    # initial position is in Anstrom
    rng = np.random.default_rng()
    #a = np.ones(10)
    a = np.ones(10) * 0.01 * 4.184 #convert to kJ/mol
    b = rng.uniform(initial_position-0.2, initial_position+0.2, 10) /10 #min/max of preloaded PMg fes x-axis.
    c = rng.uniform(1, 5.0, 10) /10
    return np.concatenate((a,b,c), axis=None)
    
def DHAM_it(CV, gaussian_params, T=300, lagtime=2, numbins=num_bins, prop_index = None):
    """
    intput:
    CV: the collective variable we are interested in. Na-Cl distance.
    gaussian_params: the parameters of bias potential. (in our case the 10-gaussian params)
     format: (a,b,c)
    T: temperature 300

    output:
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = DHAM(gaussian_params)
    d.setup(CV, T, prop_index)

    d.lagtime = lagtime
    d.numbins = numbins #num of bins, arbitrary.
    results = d.run(biased = True, plot=False) #result is [mU2, MM]
    return results

def propagate(context, gaussian_params, prop_index, PMg_dist, time_tag, steps=10000, dcdfreq=100,  platform=platform, stepsize=stepsize, num_bins=num_bins, reach = None):
    """
    propagate the system for a number of steps.
    we have to start from loading the molecule, FF
    then we define global params for bias (random guess around starting position if first propagation)
    
    each time, we generate new bias
    and propagate the system for a number of steps.
    use the partial free energy landscape to generate next bias params.
    repeat.
    """
    
    
    file_handle = open(f"trajectory/explore_traj/{time_tag}_PMg_exploring_traj_{prop_index}.dcd", 'bw')
    dcd_file = omm_app.dcdfile.DCDFile(file_handle, psf.topology, dt = stepsize)

    dist = []

    progress_bar = tqdm(total = steps, desc="MD progress")

    for i_step in range(int(steps)):

        #update progress bar.
        if i_step % (steps//10) == 0 or i_step == steps-1:
            progress_bar.update(i_step - progress_bar.n)


        integrator.step(1) #advance dcd freq steps and stop to record.
        state = context.getState(getPositions=True)
        
        pos1 = state.getPositions(asNumpy=True)[7799] #Mg
        pos2 = state.getPositions(asNumpy=True)[7840] #P
        dist.append(np.linalg.norm(pos1-pos2) * 10) #convert to angstrom
        
        #every dcdfreq we save the dcd_file
        if i_step % dcdfreq == 0:
            dcd_file.writeModel(state.getPositions(asNumpy=True)) # if we write everyframe this will be slow.
    file_handle.close()

    #now we have the trajectory, we can calculate the Markov matrix.
    np.savetxt(f"distance/{time_tag}_PMg_exploring_traj_{prop_index}.txt", dist) #save the distance to a txt file. 
    
    for index_d, d in enumerate(dist):
        if d >= 8:
            print("at step: ", index_d * dcdfreq, "we reach 8A")
            reach = index_d * dcdfreq
    
    #we concatenate the new dist to the old dist.
    # PMg_dist is a list of renewed dist.
    combined_dist = np.concatenate((PMg_dist[-1], dist), axis=None)
    PMg_dist.append(combined_dist)

    F_M, MM = DHAM_it(combined_dist.reshape(-1, 1), gaussian_params, T=300, lagtime=1, numbins=num_bins, prop_index = prop_index)

    cur_pos = combined_dist[-1] #the last position of the traj. not our cur_pos is the CV distance.
    
    return cur_pos, PMg_dist, MM, reach, F_M 

def minimize(context):
    s = time.time()
    print("Setting up the simulation")

    # Minimizing step
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()

    for _ in tqdm(range(1000), desc="Minimization"):
        omm.openmm.LocalEnergyMinimizer.minimize(context, 10, 1000)
        state = context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()
    
    print("Minimization done in", time.time() - s, "seconds")
    return context, energy

def add_bias(system, gaussian_params, num_gaussian=10):
    """
    gaussian params: np.array([a,b,c])
    system: the openmm system object.
    """
    a = gaussian_params[:num_gaussian]
    b = gaussian_params[num_gaussian:2*num_gaussian]
    c = gaussian_params[2*num_gaussian:]
    potential = ' + '.join(f'a{i}*exp(-(r-b{i})^2/(2*c{i}^2))' for i in range(num_gaussian))
    print(potential)
    custom_bias = omm.CustomBondForce(potential)
    
    for i in range(num_gaussian):
        custom_bias.addGlobalParameter(f'a{i}', a[i])
        custom_bias.addGlobalParameter(f'b{i}', b[i])
        custom_bias.addGlobalParameter(f'c{i}', c[i])
        custom_bias.addBond(7799, 7840)  #Mg 7799, P 7840

    system.addForce(custom_bias)
    return system

def get_working_MM(M):
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

def get_closest_state(qspace, target_state, working_indices):
    """
    usesage: qspace = np.linspace(1.0, 8.0, 100+1)
    target_state = 6 #find the closest state to 6A.
    """

    working_states = qspace[working_indices] #the PMg distance of the working states.
    closest_state = working_states[np.argmin(np.abs(working_states - target_state))]
    return closest_state

def read_params(filename):
    parFiles = ()
    for line in open(filename, 'r'):
        if '!' in line: line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0: parFiles += ( parfile, )

    params = CharmmParameterSet( *parFiles )
    return params

###############################################
# here we start the main python process:
# propagate -> acess the Markov Matrix -> biasing -> propagate ...
###############################################

if __name__ == "__main__":

    psf = omm_app.CharmmPsfFile(psf_file)
    pdb = omm_app.PDBFile(pdb_file)

    """with open("output_files/NaCl_solvated_system", 'r') as file_handle:
        xml = file_handle.read()
    system = omm.XmlSerializer.deserialize(xml)

    """
    #forcefield = omm_app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    params = read_params('toppar.str')
    for i_sim in range(num_simulations):
        system = psf.createSystem(params, nonbondedMethod=NoCutoff, nonbondedCutoff=1*unit.nanometer, constraints=HBonds)
        platform = omm.Platform.getPlatformByName('CUDA')
        
        #### setup an OpenMM context
        integrator = omm.LangevinIntegrator(T*unit.kelvin, #Desired Integrator
                                            fricCoef/unit.picoseconds,
                                            stepsize*unit.femtoseconds)
        qspace = np.linspace(1, 8, num_bins+1) #hard coded for now.
        PMg_dist = [[]]
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        reach = None
        i = 0
        total_steps = None

        while reach is None:
            if i > max_propagation:
                print("Max propagation reached. Exiting.")
                break
            if i == 0:
                print("propagation number 0 STARTING.")
                gaussian_params = random_initial_bias(initial_position = 3.7)
                biased_system = add_bias(system, gaussian_params)

                ## construct an OpenMM context
                context = omm.Context(biased_system, integrator)   
                if minimizing:
                    context, energy = minimize(context)         #minimize the system
                else: #we need initialize the position from the pdb file.
                    context.setPositions(pdb.positions)

                ## MD run "propagation"
                cur_pos, PMg_dist, M, reach, mU2= propagate(context, 
                                                gaussian_params=gaussian_params, 
                                                PMg_dist = PMg_dist,
                                                time_tag = time_tag,
                                                prop_index=i,
                                                steps=propagation_step, 
                                                dcdfreq=dcdfreq, 
                                                platform=platform, 
                                                stepsize=stepsize,
                                                )
                matrices.append(M)

                analytic_matrix = np.linalg.matrix_power(M, 10000)
                matricies_analytic.append(analytic_matrix)

                #finding the closest element in MM to the end point. 8A in np.linspace(2.4, 9, 150+1)
                #trim the zero rows and columns markov matrix to avoid 0 rows.
                #!!!do everything in index space. !!!
                cur_pos_index = np.digitize(cur_pos, qspace) #the big index on full markov matrix.

                working_MM, working_indices = get_working_MM(M) #we call working_index the small index. its part of the full markov matrix.
                final_index = np.digitize(8, qspace) #get the big index of desired 8A PMg distance.
                farest_index = working_indices[np.argmin(np.abs(working_indices - final_index))] #get the closest to the final index in qspace.

                i += 1

            else:
                print(f"propagation number {i} STARTING.")
                #renew the gaussian params using returned MM.
                print("getting gaussian parameters")

                #here instead of the cur_pos_index, we use the average of the last 50 positions.
                #avg_last50 = np.mean(NaCl_dist[-1][-50:])
                #avg_last50_digitized = np.digitize(avg_last50, qspace)
                
                print("farest index is: ", farest_index)
                print("cur_pos_index is: ", cur_pos_index)

                gaussian_params = try_and_optim_M(working_MM, 
                                                working_indices = working_indices,
                                                num_gaussian=10, 
                                                start_state=cur_pos_index, 
                                                end_state=farest_index,
                                                plot = False,
                                                )
                #save the gaussian params to a txt file.
                np.savetxt(f"gaussian_params/{time_tag}_gaussian_params_{i}.txt", gaussian_params)

                #we use context.setParameters() to update the bias potential.
                for j in range(10):
                    context.setParameter(f'a{j}', gaussian_params[j] * 4.184) #unit in openmm is kJ/mol, the a is fitted in kcal/mol, so we multiply by 4.184
                    context.setParameter(f'b{j}', gaussian_params[j+10] / 10) #unit openmm is nm, the b is fitted in A, so we divide by 10.
                    context.setParameter(f'c{j}', gaussian_params[j+20] / 10) #same to b.
                
                #we plot the total bias.
                test_gaussian_params = []
                for j in range(10):
                    test_gaussian_params.append(context.getParameter(f'a{j}')) #we keep energy in kj/mol in plot.
                    test_gaussian_params.append(context.getParameter(f'b{j}')*10) #b, c in A. 
                    test_gaussian_params.append(context.getParameter(f'c{j}')*10)

                test_total_bias = np.zeros_like(qspace)
                for n in range(len(test_gaussian_params)//3):
                    test_total_bias += gaussian(qspace, test_gaussian_params[3*n], test_gaussian_params[3*n+1], test_gaussian_params[3*n+2])

                plt.figure()
                mU2 = mU2 * 4.184 #convert to kJ/mol
                #plt.plot(qspace[:num_bins], mU2, label="reconstructed M_FES by DHAMsym")
                plt.plot(qspace, test_total_bias, label = "total bias")
                #plot the cur_pos
                plt.axvline(x=cur_pos, color='r', linestyle='--', label="current position")
                plt.xlim(0.5, 9)
                plt.xlabel("P-Mg distance (A)")
                plt.ylabel("Free energy (kJ/mol)")
                plt.legend()
                plt.savefig(f"fig/explore/{time_tag}_bias_{i}.png")
                plt.close()



                ## MD run "propagation"
                cur_pos, PMg_dist, M, reach, mU2= propagate(context, 
                                                gaussian_params=gaussian_params, 
                                                PMg_dist = PMg_dist,
                                                time_tag = time_tag,
                                                prop_index=i,
                                                steps=propagation_step, 
                                                dcdfreq=dcdfreq, 
                                                platform=platform, 
                                                stepsize=stepsize,
                                                )
                matrices.append(M)
                analytic_matrix = np.linalg.matrix_power(M, 10000)
                matricies_analytic.append(analytic_matrix)
                cur_pos_index = np.digitize(cur_pos, qspace) #update cur_pos_index

                working_MM, working_indices = get_working_MM(M)
                farest_index = working_indices[np.argmin(np.abs(working_indices - final_index))] #get the closest to the final index
                i += 1
        
        #we have reach != None, as a index.
        #calculate the total steps used.
        total_steps = i*propagation_step + reach*dcdfreq #reach is the index of the last step in the last propagation.
        print("total steps used: ", total_steps)

        with open("total_steps_mfpt.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([total_steps])
        #save the NaCl_dist
        np.savetxt(f"distance/{time_tag}_NaCl_exploring_traj.txt", PMg_dist[-1])

    
    


            
            
    
    
    
    
    
    

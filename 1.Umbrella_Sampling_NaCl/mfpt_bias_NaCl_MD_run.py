import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import openmm.app as omm_app
import openmm as omm
#import openmm.unit as unit
from openmm.unit import *
from tqdm import tqdm
import mdtraj
from util import *
import os
import csv

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
        print("metircs written")



if len(sys.argv) != 3:
    print("Expected 2 arguments: run_type, run_number")
    sys.exit(1)

run_type = sys.argv[1]
run_number = int(sys.argv[2])


print(f"Run Type: {run_type}, Run Number: {run_number}")





psf_file = 'toppar/step3_input.psf' # Path
pdb_file = 'toppar/step3_input25A.pdb' # Path
psf = omm_app.CharmmPsfFile(psf_file)
pdb = omm_app.PDBFile(pdb_file)



## reading in the OpenMM system we created
with open(f"output_files/NaCl_solvated_system_mfpt_bias_{run_number}", 'r') as file_handle:
    xml = file_handle.read()
system = omm.XmlSerializer.deserialize(xml)

# Setting up the platform

# platform = omm.Platform.getPlatformByName('CUDA')     # If you have GPU try this
# platform = omm.Platform.getPlatformByName('OpenCL')   # Or this one, if the preivous does not work
platform = omm.Platform.getPlatformByName('CPU')      # Try first using CUDA or OpenCL it is way faster


#### setup an OpenMM context

T = 298.15      # temperature in K
fricCoef = 10   # friction coefficient in 1/ps
stepsize = 2    # integration step size in fs
dcdfreq = 100   # save coordinates at every 100 step
steps = 100000  # total steps

integrator = omm.LangevinIntegrator(T*kelvin, #Desired Integrator
                                    fricCoef/picoseconds,
                                    stepsize*femtoseconds) 
## construct an OpenMM context
context = omm.Context(system, integrator)   # you may pass platform as a third positional argument


st = time.time()

s = time.time()
print("Setting up the simulation")

#note in our case all gaussian parameters has been determined.
# Minimizing step
context.setPositions(pdb.positions)
state = context.getState(getEnergy = True)
energy = state.getPotentialEnergy()

for i in tqdm(range(50), desc="Minimization"):
    omm.openmm.LocalEnergyMinimizer.minimize(context, 1, 20)
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()

print("Minimization done in", time.time() - s, "seconds")
s = time.time()

NaCl_xdist = []
# Sampling production. trajectories are saved in dcd files
file_handle = open(f"trajectories_bias/mfpt_bias_traj_{run_number}.dcd", 'bw')
dcd_file = omm_app.dcdfile.DCDFile(file_handle, psf.topology, dt = stepsize)
for i in tqdm(range(int(steps/dcdfreq)), desc="Production"):
    integrator.step(dcdfreq)
    state = context.getState(getPositions = True)
    NaCl_xdist.append(state.getPositions(asNumpy=True)[0]._value)
    positions = state.getPositions()
    dcd_file.writeModel(positions)
file_handle.close()
print("Production Run done in:", str(time.time() - s), "seconds")


#load the trajectory using mdtraj.
traj = mdtraj.load_dcd(f"trajectories_bias/mfpt_bias_traj_{run_number}.dcd", top=pdb_file)
dist = mdtraj.compute_distances(traj, [[0, 1]]) # Calculate distance between Na and Cl
print("going through dist")
for final_frame, i_dist in enumerate(dist):
    
    if i_dist > 0.7:
        print("reached 7: ", i_dist)
        time_ps = (final_frame *100 *2)/1000
        print("updating metrics") 
        update_metrics(run_type, run_number, final_frame, time_ps)
        sys.exit(1)

x = np.linspace(0.2, 0.9, 100)
#get the histogram, normalized.
hist= np.histogram(dist, x)[0]
hist = hist / np.sum(hist)

#plot the analytical boltzmann distribution

plt.plot(np.linspace(0.2, 0.9, 99), hist)
plt.xlabel("Distance (A)")
plt.ylabel("Probability")
plt.show()

print("All Done!")

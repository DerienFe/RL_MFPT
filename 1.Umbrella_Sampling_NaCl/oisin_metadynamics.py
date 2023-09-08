import openmm as omm
from openmm import LangevinIntegrator
from openmm.app import *
from openmm.unit import *
import tqdm
import matplotlib.pyplot as plt
import numpy as np 
import os
import csv
total_stpes = 100000
record_freq=50

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



psf_file = 'toppar/step3_input.psf' # Path
pdb_file = 'toppar/step3_input25A.pdb' # Path
psf = omm.app.CharmmPsfFile(psf_file)
pdb = omm.app.PDBFile(pdb_file)
distances = []
bias_energies = []

#params = omm_app.CharmmParameterSet('toppar/toppar_water_ions.str') #old way.
## Create an OpenMM system
##system = psf.createSystem(params) #old way.
from openmm.app import *
from openmm.unit import *
forcefield = omm.app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')

system = forcefield.createSystem(pdb.topology,
                                 nonbondedCutoff=1.0*nanometers, 
                                 constraints=omm.app.HBonds)


# Define the equation
#bias_bond = omm.CustomBondForce("0.5*k*(r-r0)^2") 

# Initialize parameters, these will be later set
#bias_bond.addGlobalParameter("k", 1.0)  # Force constant in kJ/(mol*nm^2)
#bias_bond.addGlobalParameter("r0", 0.0) # Optimal value of the distance in nm

bias_bond = omm.CustomBondForce("r")

# Define the bond
bias_bond.addBond(0, 1) # Here you would have to put the atoms you will be adding the bond to
x = BiasVariable(bias_bond, 2.0, 10, 0.5, True)

meta = Metadynamics(system, [x], 300*kelvin, 8.0,
                    1.0*kilojoules_per_mole, 1000)

platform = omm.Platform.getPlatformByName('CUDA')

sim = Simulation(pdb.topology, system, LangevinIntegrator(300*kelvin, 1/picosecond,
                 0.002*picoseconds), platform=platform) 
sim.context.setPositions(pdb.positions)

for i in tqdm.tqdm(range(int(total_stpes/record_freq))):
    meta.step(sim, record_freq)
    dist = meta.getCollectiveVariables(sim)
    distance = dist[0]
    if distance >= 0.7: 
        print("reached 7: ", dist)
        
        # Calculate the total simulation time in picoseconds.
        time_ns = (i * 50 * 2)/1000

        # Convert the simulation time to nanoseconds.
        
        print(f"Simulation time when distance reached 7: {time_ns} ns")
        run_type = "Metadynamics"
        run_number = 19
        final_frame = i*50
        print("updating metrics") 
        print(run_type, run_number, final_frame, time_ns)
        update_metrics(run_type, run_number, final_frame, time_ns)
        
        sys.exit(1)

    distances.append(dist[0])
    state = sim.context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    bias_energy = potential_energy.value_in_unit(kilocalorie_per_mole)
    bias_energies.append(bias_energy)

# Generate an array of positions, corresponding to the index in the 'distances' array
distances = np.array(distances)
positions = np.arange(len(distances))

# Create a plot
plt.figure(figsize=(10, 6))

# Plot the distances against their positions in the array
plt.scatter(positions, distances, c='blue', label='Distances')

# Add labels and title
plt.xlabel('Position in Array')
plt.ylabel('Distance')
plt.title('Distances vs Position in Array')
plt.grid(True)

#print(distances)
print(len(bias_energies))
plt.xlabel('Na-Cl distance (nm)')
plt.plot(distances, bias_energies)
plt.ylabel('Free energy (kJ/mol)')
plt.show()

# Plot histogram for distances
plt.figure()
plt.hist(distances, bins=50, density=True, alpha=0.7, color='g')
plt.title('Histogram of Distances')
plt.xlabel('Distance (nm)')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for bias energies
plt.figure()
plt.hist(bias_energies, bins=50, density=True, alpha=0.7, color='r')
plt.title('Histogram of Bias Energies')
plt.xlabel('Energy (kcal/mol)')
plt.ylabel('Frequency')
plt.show()
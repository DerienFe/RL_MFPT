import openmm as omm
from openmm import LangevinIntegrator
from openmm.app import *
from openmm.unit import *
import tqdm
import matplotlib.pyplot as plt




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

platform = omm.Platform.getPlatformByName('CPU')

sim = Simulation(pdb.topology, system, LangevinIntegrator(300*kelvin, 1/picosecond,
                 0.002*picoseconds), platform=platform) 
sim.context.setPositions(pdb.positions)

for i in tqdm.tqdm(range(50)):
    meta.step(sim, 100)
    dist = meta.getCollectiveVariables(sim)
    distances.append(dist[0])
    state = sim.context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    bias_energy = potential_energy.value_in_unit(kilocalorie_per_mole)
    bias_energies.append(bias_energy)



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
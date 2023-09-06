from simtk.openmm.app import PDBFile, CharmmPsfFile, Metadynamics, BiasVariable, Simulation
from simtk.openmm import LangevinIntegrator, Platform, CustomCVForce
from simtk.unit import kelvin, picosecond, femtosecond, nanometer, kilocalories_per_mole

# Read input files
pdb = PDBFile('input.pdb')
psf = CharmmPsfFile('input.psf')

# Define the system
system = psf.createSystem(nonbondedMethod=app.PME)

# Define the integrator
integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 2.0*femtosecond)

# Create simulation
simulation = Simulation(psf.topology, system, integrator)

# Set initial positions from PDB file
simulation.context.setPositions(pdb.positions)

# Collective variable: distance between Na (index 0) and Cl (index 1)
def distance(simulation):
    return CustomCVForce.distance(simulation, 0, 1)

distance_variable = BiasVariable(distance, 0.25*nanometer, 0.7*nanometer, 100)

# Metadynamics parameters
meta = Metadynamics(system, [distance_variable], 300*kelvin, 5.0,
                    0.1*kilocalories_per_mole, 100, saveFrequency=1000)

# Add metadynamics forces to the system
meta.addForceTo(system)

# Run the metadynamics simulation
while True:
    meta.step(simulation, 1000)
    current_distance = distance_variable._value(simulation)
    print(f'Current Na-Cl distance: {current_distance}')
    if current_distance >= 0.7*nanometer:
        print('Target distance reached.')
        break

# Save results if needed
with open('free_energy.txt', 'w') as f:
    for value in meta.getFreeEnergy():
        f.write(f'{value}\n')

from openmm import LangevinIntegrator, Platform, CustomCVForce, CustomBondForce
from openmm.app import Metadynamics, BiasVariable, CharmmPsfFile, PDBFile, ForceField
from openmm.unit import kelvin, picosecond, nanometers, kilojoules_per_mole

# Load your NaCl system from PSF and PDB files
psf_file = 'toppar/step3_input.psf'  # Replace with your actual path
pdb_file = 'toppar/step3_input.pdb'  # Replace with your actual path

psf = CharmmPsfFile(psf_file)
pdb = PDBFile(pdb_file)

forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')

system = forcefield.createSystem(psf.topology,
                                 nonbondedCutoff=1.0*nanometers,
                                 constraints=None)

# Define your collective variable with a CustomBondForce
cv_force = CustomBondForce("r")
cv_force.addBond(0, 1, [])  # Assuming Na and Cl atoms are 0 and 1, adjust as needed

# Wrap it in a CustomCVForce
cv = CustomCVForce(cv_force)

# Add the CV force to the system
system.addForce(cv)

# Define BiasVariable for Metadynamics
bias_variable = BiasVariable(cv, 2.5*nanometers, 7.0*nanometers, 0.1*nanometers, False)

# Create the Metadynamics object
meta = Metadynamics(system, [bias_variable], 300.0*kelvin, 2.0, 1.0*kilojoules_per_mole, 100)

# Setup simulation
integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)
platform = Platform.getPlatformByName('CPU')  # or 'CUDA', 'OpenCL' based on your system
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

# Run the simulation
meta.step(simulation, 5000)  # 5000 steps for demonstration, you'd typically run much longer

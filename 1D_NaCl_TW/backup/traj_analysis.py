import mdtraj as md
from tqdm import tqdm

# Load the trajectory
topology = "toppar/step3_input25A.pdb"  # Change this to your topology file
trajectory = "/Users/oisinmoriarty/Documents/trajectories/mfpt_bias_traj_3.dcd"  # Change this to your DCD file
traj = md.load(trajectory, top=topology)

# Find the indices of sodium and chloride ions
# Assume their names in the topology are 'Na' and 'Cl', respectively.
na_indices = [atom.index for atom in traj.topology.atoms if atom.element.symbol == 'Na']
cl_indices = [atom.index for atom in traj.topology.atoms if atom.element.symbol == 'Cl']

# Initialize variables to hold frames where the distance is > 7
frames_with_large_distance = []

# Calculate distances for each frame
for i in tqdm(range(len(traj)), desc="Analyzing frames"):
    frame = traj.xyz[i]
    for na_index in na_indices:
        for cl_index in cl_indices:
            distance = md.compute_distances(traj[i:i+1], [[na_index, cl_index]])[0][0] * 10  # convert to Angstrom
            if distance > 7:
                print(f"Distance greater than 7 Angstrom found in frame {i} between Na index {na_index} and Cl index {cl_index}. Distance: {distance} Angstrom")
                frames_with_large_distance.append((i, na_index, cl_index, distance))

# If you wish to do something with frames_with_large_distance, you can do so here.

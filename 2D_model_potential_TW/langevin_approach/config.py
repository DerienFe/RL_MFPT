#this is a config file for langevin_sim series.

import time
from openmm import unit
import openmm

from openmm.unit import Quantity
from openmm import Vec3

sim_steps = int(1e4)
pbc = True
time_tag = time.strftime("%Y%m%d-%H%M%S")
amp = 10 #for amp applied on fes. note the gaussian parameters for fes is normalized.

propagation_step = 1000
stepsize = 0.002 * unit.picoseconds
num_bins = 20 #used to discretize the traj, and used in the DHAM.
dcdfreq = 100

platform = openmm.Platform.getPlatformByName('CUDA')
#platform = openmm.Platform.getPlatformByName('CPU')

num_gaussian = 20 #number of gaussians used to placing the bias.



#starting state (as in coordinate space, from 0 to 2pi.)
start = Quantity(value = [Vec3(1.0,0.1,0.0)], unit = unit.nanometers)
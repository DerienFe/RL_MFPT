#this is a python script do the bar plot for different algo used exploring the 2D.
# including: unbiased, single_K_optimized, exploring_M_optimized

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})


#load dthe data.
#note, convert all reaching time to standard time. some of the time recorded is 0.01 for example.

unbiased_time_mean, unbiased_time_std = np.loadtxt("./data/unbiased_mfpt_mean_std_(14, 14)_(4, 6).txt")




plt.figure(figsize=(8, 6))
plt.bar([1], unbiased_time_mean, yerr=unbiased_time_std, color='b', width=0.5, label='unbiased')
plt.show()




print("all done！")
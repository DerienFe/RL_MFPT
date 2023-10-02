#this is a python script fit some gaussians to the dialanine fes.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image

# Define a single 2D Gaussian
def gaussian_2D(params, x, y):
    A, x0, y0, sigma_x, sigma_y = params
    return A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

# Define the sum of Gaussians
def sum_of_gaussians(params, x, y, n_gaussians, N=100):
    total = np.zeros((N,N))
    for i in range(n_gaussians):
        total += gaussian_2D(params[i*5:i*5+5], x, y)
    return total

# Define the error function to be minimized
def error_function(params, x, y, fes, n_gaussians):
    fes_dim = fes.shape[0]
    diff = fes - sum_of_gaussians(params, x, y, n_gaussians, N = fes_dim)
    return np.sum(diff**2)

# Assume x_data, y_data, and fes_data are the 2D grid and FES values
# Let's fit using 5 Gaussians for illustration
n_gaussians = 10
img = Image.open("./fes_digitize.png")
img = np.array(img)

img_greyscale = 0.8 * img[:,:,0] - 0.15 * img[:,:,1] - 0.2 * img[:,:,2]
img = img_greyscale
img = img/np.max(img)
img = img - np.min(img)

#get img square.
img = img[0:img.shape[0], 0:img.shape[0]]

#note fes is normalized.
#now we fit it with gaussians
#we cast this into x,y coordinates [-pi, pi] x [-pi, pi]
x, y = np.meshgrid(np.linspace(-np.pi, np.pi, img.shape[0]), np.linspace(-np.pi, np.pi, img.shape[0]))
#fit the gaussians

#we distribute initial guess randomly over the grid.
initial_guess = np.zeros(n_gaussians*5)
initial_guess[0::5] = np.random.uniform(low=0.8, high=1, size=n_gaussians) #amplitude
initial_guess[1::5] = np.random.uniform(low=-np.pi, high=np.pi, size=n_gaussians) #x0
initial_guess[2::5] = np.random.uniform(low=-np.pi, high=np.pi, size=n_gaussians) #y0
initial_guess[3::5] = np.random.uniform(low=0.5, high=5, size=n_gaussians) #sigma_x
initial_guess[4::5] = np.random.uniform(low=0.5, high=5, size=n_gaussians) #sigma_y

result = minimize(error_function, initial_guess, args=(x, y, img, n_gaussians),tol=1e-0)

#visualize the result
reconstructed = sum_of_gaussians(result.x, x, y, n_gaussians, N=x.shape[0])
plt.figure()
plt.imshow(reconstructed, origin='lower')
plt.show()

#save the result
np.savetxt("./fes_digitize_gauss_params.txt", result.x)

print("All done")
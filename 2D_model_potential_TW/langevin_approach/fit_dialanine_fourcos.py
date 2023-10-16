#this is a python script fit fourier cosine set to the dialanine fes.
# because we wanted to represenet the dialanine phi/psi fes as a sum of fourier cosine set.
# on -pi and pi, the periodicity of the fes should be 2pi. L = pi.
# the fes is usually neither even nor odd. so we use the following form:
# f(x) = a0 + sum_{n=1}^{N} a_n cos(n pi x / L) + b_n sin(n pi x / L)
#  note this is only for x, we do the same for y.
#  thus f(x,y) = a0 + sum_m sum_n (
#                   a_mn cos(m pi x / L) cos(n pi y / L) +
#                   b_mn sin(m pi x / L) cos(n pi y / L) +
#                   c_mn cos(m pi x / L) sin(n pi y / L) +
#                   d_mn sin(m pi x / L) sin(n pi y / L)
#                   )
# where a0, a_mn, b_mn, c_mn, d_mn are the parameters to be fitted. given L = pi.
# note the full set is too expensive, we use the cos*cos terem only, and we phase shift it -pi to adapt our (-pi, pi) range.
#  this will give us:
#  f(x,y) = a0 + sum_m sum_n (
#                 a_mn cos(m pi x / L) cos(n pi y / L))
# fitting parameters: a0, a_mn
# #change the periodicity.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from PIL import Image

# Define a single 2D cosine fourier set
def fourier_sincos_full(params, x, y, m, n):
    a0 = params[0]
    a = params[1:m*n+1]
    b = params[m*n+1:2*m*n+1]
    c = params[2*m*n+1:3*m*n+1]
    d = params[3*m*n+1:4*m*n+1]
    total = a0
    for i in range(m):
        for j in range(n):
            total += a[i*m+j] * np.cos((i+1) * np.pi * x / np.pi) * np.cos((j+1) * np.pi * y / np.pi) 
            total += b[i*m+j] * np.sin((i+1) * np.pi * x / np.pi) * np.cos((j+1) * np.pi * y / np.pi)
            total += c[i*m+j] * np.cos((i+1) * np.pi * x / np.pi) * np.sin((j+1) * np.pi * y / np.pi)
            total += d[i*m+j] * np.sin((i+1) * np.pi * x / np.pi) * np.sin((j+1) * np.pi * y / np.pi)
    return total

def fourier_cos_partial(params, x, y, m, n):
    a0 = params[0]
    a = params[1:m*n+1]
    total = a0
    for i in range(m):
        for j in range(n):
            total += a[i*m+j] * np.cos((i+1) * np.pi * x / np.pi) * np.cos((j+1) * np.pi * y / np.pi)
    return total

# Define the error function to be minimized
def error_function(params, x, y, fes, m, n):
    diff = fes - fourier_sincos_full(params, x, y, m, n) #fourier_sincos_full(params, x, y, m, n)
    return np.sum(diff**2)

def residuals(params, x, y, fes, m, n):
    return (fes - fourier_sincos_full(params, x, y, m, n)).ravel()

# Assume x_data, y_data, and fes_data are the 2D grid and FES values
m = 3 #order of the fourier set
n = 3
test = False
img = Image.open("./fes_digitize.png")
img = np.array(img)
target_minimize = True
target_leastsq = False
img_greyscale =  0.6 * img[:,:,0] + 0.4 * img[:,:,1] + 0.11 * img[:,:,2] #0.8 * img[:,:,0] - 0.15 * img[:,:,1] - 0.2 * img[:,:,2]
img = img_greyscale
img = img - np.min(img)
img = img/np.max(img)

#get img square.
img = img[0:img.shape[0], 0:img.shape[0]]

#the image is on -pi to pi, we shift it to 0 to 2pi
img = np.roll(img, int(img.shape[0]/2), axis=0)
img = np.roll(img, int(img.shape[1]/2), axis=1)

plt.figure()
plt.imshow(img, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=1.2)
plt.savefig("./test.png")
plt.close()


#note fes is normalized.
#we cast this into x,y coordinates [-pi, pi] x [-pi, pi]
#x, y = np.meshgrid(np.linspace(-np.pi, np.pi, img.shape[0]), np.linspace(-np.pi, np.pi, img.shape[0]))
if target_minimize:
    x, y = np.meshgrid(np.linspace(0, 2*np.pi, img.shape[0]), np.linspace(0, 2* np.pi, img.shape[0]))

    #we distribute initial guess randomly over the grid.
    """initial_guess = np.zeros(4*m*n+1)
    initial_guess[0] = np.random.uniform(low=0.8, high=1, size=1) #a0
    initial_guess[1:m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #a
    initial_guess[m*n+1:2*m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #b
    initial_guess[2*m*n+1:3*m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #c
    initial_guess[3*m*n+1:4*m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #d
"""
    initial_guess = np.zeros(4*m*n+1)
    initial_guess[0] = 0.5
    initial_guess[1:m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #a
    initial_guess[m*n+1:2*m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #b
    initial_guess[2*m*n+1:3*m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #c
    initial_guess[3*m*n+1:4*m*n+1] = np.random.uniform(low=-1, high=1, size=m*n) #d

    result = minimize(error_function, initial_guess, args=(x, y, img, m, n), method='Nelder-Mead', tol=1e-0)
    np.savetxt("./fes_digitize_fourier_params.txt", result.x)

    #visualize the result
    reconstructed = fourier_cos_partial(result.x, x, y, m, n)
    plt.figure()
    plt.imshow(reconstructed, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi])
    plt.savefig("./fes_digitize_fourier.png")
    plt.close()

    print("All done")
if target_leastsq:
    from scipy.linalg import leastsq
    params_out, conv_x, infodict, mesg, ier = leastsq(residuals, initial_guess, args=(x, y, img, m, n), full_output=True)
    np.savetxt("./fes_digitize_fourier_params_ls.txt", params_out)

    #visualize
    reconstructed = fourier_sincos_full(params_out, x, y, m, n)
    plt.figure()
    plt.imshow(reconstructed, cmap="coolwarm", extent=[-np.pi, np.pi,-np.pi, np.pi])
    plt.savefig("./fes_digitize_fourier_ls.png")
    
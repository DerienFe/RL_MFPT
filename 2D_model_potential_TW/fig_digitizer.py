import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def create_K_png(N, img_path = "./fes_digitize.png", kT = 0.5981):
    """
    read in the png, digitize it, create a fes based on it.
        the created fes is [N,N] in shape.
        we made sure the fes is normalized to min/max of 0/1.
        and then apply the amplitude of A = 4 to it.
    and then create the K matrix from the fes. (2D)
    """
    amp = 6

    img = Image.open(img_path)
    img = np.array(img)

    img_greyscale = 0.2989 * img[:,:,0] - 0.1140 * img[:,:,2] + 0.5870 * img[:,:,1]
    img = img_greyscale
    img = img/np.max(img)
    img = img - np.min(img)

    #we only take points in image every ? steps so it has [N,N] shape.
    img = img[::int(img.shape[0]/N), ::int(img.shape[1]/N)]
    plt.imshow(img)
    plt.show()
    Z = img

    #now we create the K matrix.
    K = np.zeros((N*N, N*N))
    for i in range(N):
        for j in range(N):
            index = np.ravel_multi_index((i,j), (N,N), order='C') # flatten 2D indices to 1D
            if i < N - 1: # Transition rates between vertically adjacent cells
                index_down = np.ravel_multi_index((i+1,j), (N,N), order='C') 
                delta_z = Z[i+1,j] - Z[i,j]
                K[index, index_down] = amp * np.exp(delta_z / (2 * kT))
                K[index_down, index] = amp * np.exp(-delta_z / (2 * kT))
            if j < N - 1: # Transition rates between horizontally adjacent cells
                index_right = np.ravel_multi_index((i,j+1), (N,N), order='C')
                delta_z = Z[i,j+1] - Z[i,j]
                K[index, index_right] = amp * np.exp(delta_z / (2 * kT))
                K[index_right, index] = amp * np.exp(-delta_z / (2 * kT))
    
    # Filling diagonal elements with negative sum of rest of row
    for i in range(N*N):
        K[i, i] = -np.sum(K[:,i])

    return K



K = create_K_png(40)

print("done")
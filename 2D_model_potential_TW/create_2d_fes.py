import numpy as np
from PIL import Image

def create_K_png(N, img_path = "./fes_digitize.png", kT = 0.5981, amp = 7):
    """
    read in the png, digitize it, create a fes based on it.
        the created fes is [N,N] in shape.
        we made sure the fes is normalized to min/max of 0/1.
        and then apply the amplitude of A = 4 to it.
    and then create the K matrix from the fes. (2D)
    """

    img = Image.open("./fes_digitize.png")
    img = np.array(img)

    img_greyscale = 0.8 * img[:,:,0] - 0.15 * img[:,:,1] - 0.2 * img[:,:,2]
    img = img_greyscale
    img = img/np.max(img)
    img = img - np.min(img)
    x, y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N))
    #plt.imshow(img, cmap="coolwarm")
    #plt.savefig("./figs/unbiased.png", dpi=600)
    #plt.show()
    #we only take points in image every ? steps so it has [N,N] shape.
    img = img[::int(img.shape[0]/N), ::int(img.shape[1]/N)]
    #plt.imshow(img, cmap="coolwarm")
    #plt.show()
    Z = img * amp

    #now we create the K matrix.
    K = np.zeros((N*N, N*N))
    for i in range(N):
        for j in range(N):
            index = np.ravel_multi_index((i,j), (N,N), order='C') # flatten 2D indices to 1D
            if i < N - 1: # Transition rates between vertically adjacent cells
                index_down = np.ravel_multi_index((i+1,j), (N,N), order='C') 
                delta_z = Z[i+1,j] - Z[i,j]
                K[index, index_down] = np.exp(delta_z / (2 * kT))
                K[index_down, index] = np.exp(-delta_z / (2 * kT))
            if j < N - 1: # Transition rates between horizontally adjacent cells
                index_right = np.ravel_multi_index((i,j+1), (N,N), order='C')
                delta_z = Z[i,j+1] - Z[i,j]
                K[index, index_right] = np.exp(delta_z / (2 * kT))
                K[index_right, index] = np.exp(-delta_z / (2 * kT))
    
    # Filling diagonal elements with negative sum of rest of row
    for i in range(N*N):
        K[i, i] = -np.sum(K[:,i])

    return K

def create_K_2D(N, kT):
    """
    N is the total state number
    kT is the thermal energy

    here we create a 2D potential surface. [100,100]
    """
    gamma = 0.01
    x, y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N))
    barrier_height = 0.7
    Z = 5 - barrier_height * np.log((np.exp(-(x+2)**2 - 3*(y+2)**2))/gamma + 
                                    (np.exp(-5*(x-2)**2 - (y-2)**2))/gamma +
                                    (np.exp(-6*(x-2)**2 - 5*(y+2)**2))/gamma+
                                    (np.exp(-3*(x+2)**2 - (y-2)**2))/gamma 
                                    )
    #zero the Z
    Z = Z - np.min(Z)
    amp = 4

    K = np.zeros((N*N, N*N)) # we keep K flat. easier to do transpose etc.

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

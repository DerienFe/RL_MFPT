import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import matplotlib.animation as animation


# Example list of matrices
Ms = [np.random.rand(10, 10) for _ in range(20)]





# Animation function    
def animate(i):
    plt.imshow(Ms[i], cmap='hot')

# Create animation
fig = plt.figure()
anim = animation.FuncAnimation(fig, animate, frames=len(Ms), interval=500) 

# Save animation
anim.save('markov_animation.gif', writer='imagemagick') 

plt.show()
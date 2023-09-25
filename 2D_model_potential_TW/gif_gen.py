import imageio
import os

path = './figs/prod_figs/movie_20230925-131244/'

# Get first frame in path unbiased.png
first_img = path + 'unbiased.png'
#get the rest of png in pwd not first_img.
file_names = sorted((fn for fn in os.listdir(path) if fn.endswith('.png') and fn!=first_img))

# Create a writer object
writer = imageio.get_writer('./figs/prod_figs/movie_20230925-131244/movie.gif', fps=2)
writer.append_data(imageio.imread(first_img))
# Add images to the writer object
for file_name in file_names:
    writer.append_data(imageio.imread(os.path.join(path, file_name)))

# Close the writer object
writer.close()

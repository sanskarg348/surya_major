import numpy as np

from PIL import Image
from numpy import asarray

# load the image and convert into
# numpy array
img = Image.open('img.png')

# asarray() class is used to convert
# PIL images into NumPy arrays
numpydata = asarray(img)

np.save('n.npy',numpydata)





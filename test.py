import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# load image

image = plt.imread('image.jpg')

r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
image = 0.2989 * r + 0.5870 * g + 0.1140 * b

sobel_h = ndimage.sobel(image, 0) # horizontal
sobel_v = ndimage.sobel(image, 1) # vertical

theta = np.arctan2(sobel_v, sobel_h)
theta

absTheta = np.abs(theta) / np.pi

fig, ax = plt.subplots()
ax.imshow(absTheta, cmap='grey')
plt.show()
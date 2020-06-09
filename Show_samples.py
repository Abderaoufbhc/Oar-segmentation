
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave, imshow


x_train = np.load('dataset_Heart/x_train.npy')
y_train = np.load('dataset_Heart/y_train.npy')
x_val = np.load('dataset_Heart/x_val.npy')
y_val = np.load('dataset_Heart/y_val.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
# Show some  random image 
image_train=x_train[177]
mask_train=y_train[177]
image_val=x_val[62]
mask_val=y_val[62]
imshow(image_train.squeeze(), cmap='gray')
plt.show()
imshow(mask_train.squeeze(), cmap='gray')
plt.show()
imshow(image_val.squeeze(), cmap='gray')
plt.show()
imshow(mask_val.squeeze(), cmap='gray')
plt.show()

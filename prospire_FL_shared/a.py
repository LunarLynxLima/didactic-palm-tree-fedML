import numpy as np
n=100
train_x_images, train_y_images = np.load('train_x_images_rti.npy', 'r')[:min(n, 70)], np.load('train_y_images_rti.npy', 'r')[:min(n, 70)]
test_x_images, test_y_images = np.load('test_x_images_rti.npy', 'r'), np.load('test_y_images_rti.npy', 'r')
print('train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape', train_x_images.shape, train_y_images.shape, test_x_images.shape, test_y_images.shape)
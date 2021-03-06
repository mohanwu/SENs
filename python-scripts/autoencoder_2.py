'''
Mostly code from
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''

from __future__ import print_function
import os
import numpy as np

import crop_encoder_data
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, Input, Conv2D, Dropout
from keras import regularizers
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from PIL import Image
parent_dir = os.path.dirname(os.getcwd())

batch_size = 50
nb_classes = 8
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 128, 128
crop_rows,crop_cols = 32,32
# number of convolutional filters to use
nb_filters = 16
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets

X_train, X_crop_train , X_test , X_crop_test = crop_encoder_data.load_data(0.5)

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_crop_train = X_crop_train.reshape(X_train.shape[0], 1, crop_rows, crop_cols)
    X_crop_test = X_crop_test.reshape(X_test.shape[0], 1, crop_rows, crop_cols)
    input_shape = (1, img_rows, img_cols)
else:
    print("DOEST HIS RUN")
    X_train = X_train.reshape(X_train.shape[0], img_rows,img_cols,1)
    X_test = X_test.reshape(X_test.shape[0], img_rows,img_cols,1)
    X_crop_train = X_crop_train.reshape(X_train.shape[0], crop_rows*crop_cols)
    X_crop_test = X_crop_test.reshape(X_test.shape[0], crop_rows*crop_cols)
    input_shape = (img_rows,img_cols,1)

X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255

X_crop_train = X_crop_train.astype('float32')
X_crop_train /= 255
X_crop_test = X_crop_test.astype('float32')
X_crop_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print('X_train shape:', X_crop_train.shape)
print(X_crop_train.shape[0], 'train samples')

nb_filters = 8
kernel_size = (3,3)
pool_size = (2,2)

input_img = Input(shape=(img_rows,img_cols,1))
encoded = Conv2D(nb_filters*2,kernel_size[0],kernel_size[1],
            border_mode="same",activation='relu',
            W_regularizer=regularizers.l1(0.1))(input_img)
encoded = BatchNormalization(mode=2)(encoded)
encoded = Conv2D(nb_filters,kernel_size[0],kernel_size[1],
            border_mode="same",activation='relu',
            W_regularizer=regularizers.l1(0.1))(encoded)
encoded = BatchNormalization(mode=2)(encoded)
encoded = Dropout(0.2)(encoded)
encoded = MaxPooling2D(pool_size=(2,2))(encoded)
encoded = Conv2D(nb_filters,kernel_size[0],kernel_size[1],
            border_mode="same",activation='relu',
            W_regularizer=regularizers.l1(0.1))(encoded)
encoded = BatchNormalization(mode=2)(encoded)
encoded = MaxPooling2D(pool_size=(2,2))(encoded)
encoded = Flatten()(encoded)
#Dont put regularizers on last layer
encoded = Dense(1024, activation='sigmoid')(encoded)

autoencoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
# at this point the representation is (32*32, 1)

if(os.path.isfile("cnn_autoencoder.h5") == True):
    del autoencoder
    autoencoder = load_model('cnn_autoencoder.h5')


if __name__ == "__main__":
    autoencoder.fit(X_train, X_crop_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1,
              callbacks=[TensorBoard(log_dir='/tmp/autoencoder')],
              #validation_data=(X_test, X_crop_test)
              )

    autoencoder.save("cnn_autoencoder.h5",overwrite=True)
    print("cnn_autoencoder "+"Saved model to disk")

def show_result(pred_result):
    result = pred_result.copy()
    result *= 255
    result = np.reshape(result,(result.shape[1],result.shape[1]))
    result = np.array(result,dtype='uint8')
    Image.fromarray(result).show()

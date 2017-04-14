'''
Mostly code from
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
'''

from __future__ import print_function
import argparse
import os
import numpy as np
np.random.seed(1337)  # for reproducibility
import crop_encoder_data
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Input, Conv2D, Dropout
from keras import regularizers
from keras.utils import np_utils
from keras import backend as K
from PIL import Image
from util import model_network_param
parent_dir = os.path.dirname(os.getcwd())

default_model = model_network_param(batch_size=50,
                                    nb_classes=8,
                                    nb_epoch=5,
                                    nb_filters=16,
                                    img_rows = 128,
                                    img_cols=128,
                                    crop_row=64,
                                    crop_cols=64,
                                    pool_size=(2,2),
                                    kernel_size=(3,3)
                                    )

class model_one():

    def __init__(self, mnp):
        self.input_img = Input(shape=(mnp.param_dict['img_rows'],
                                      mnp.param_dict['img_cols'],1))
        self.encoded = Conv2D(mnp.param_dict['nb_filters']*2,
                              mnp.param_dict['kernel_size'][0],
                              mnp.param_dict['kernel_size'][1],
                              border_mode="same",
                              activation='relu',
                              W_regularizer=regularizers.l1(0.01))(self.input_img)
        self.encoded = Conv2D(mnp.param_dict['nb_filters'],
                              mnp.param_dict['kernel_size'][0],
                              mnp.param_dict['kernel_size'][1],
                              border_mode="same",activation='relu',
                              W_regularizer=regularizers.l1(0.01))(self.encoded)
        self.encoded = Dropout(0.5)(self.encoded)
        self.encoded = MaxPooling2D(pool_size=mnp.param_dict['pool_size'])(self.encoded)
        self.encoded = Conv2D(mnp.param_dict['nb_filters'],
                              mnp.param_dict['kernel_size'][0],
                              mnp.param_dict['kernel_size'][1],
                              border_mode="same",activation='relu')(self.encoded)
        self.encoded = MaxPooling2D(pool_size=mnp.param_dict['pool_size'])(self.encoded)
        self.encoded = Flatten()(self.encoded)
        self.encoded = Dense(1024, activation='sigmoid')(self.encoded)

        self.autoencoder = Model(self.input_img, self.encoded)
        self.autoencoder.compile(optimizer='adam', loss='mean_absolute_error')

        if(os.path.isfile(mnp.param_dict['weights']) == True):
            self.autoencoder.load_weights(mnp.param_dict['weights'])
# at this point the representation is (32*32, 1)

# to build on weights previously trained

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', help='batch size for model training', type=int, default=50)
    parser.add_argument('-nb_classes', help='number of classes', type=int, default=8)
    parser.add_argument('-nb_epoch', help='number of epochs for training', type=int, default=5)
    parser.add_argument('-img_rows', help='number of pixels', type=int, default=128)
    parser.add_argument('-img_cols', help='number of pixels', type=int, default=128)
    parser.add_argument('-crop_cols', help='number of pixels', type=int, default=64)
    parser.add_argument('-crop_rows', help='number of pixels', type=int, default=64)
    parser.add_argument('-pool_size_dim1', help='number of pixels', type=int, default=2)
    parser.add_argument('-pool_size_dim2', help='number of pixels', type=int, default=2)
    parser.add_argument('-kernel_size_dim1', help='number of pixels', type=int, default=3)
    parser.add_argument('-kernel_size_dim2', help='number of pixels', type=int, default=3)
    parser.add_argument('-weights', help='filename of network weights', type=str, default='test.h5')
    args = parser.parse_args()
    if args.batch_size:
        default_model.param_dict['batch_size'] = args.batch_size
    if args.nb_classes:
        default_model.param_dict['nb_classes'] = args.nb_classes
    if args.nb_epoch:
        default_model.param_dict['nb_epoch'] = args.nb_epoch
    if args.img_rows:
        default_model.param_dict['img_rows'] = args.img_rows
    if args.img_cols:
        default_model.param_dict['img_cols'] = args.img_cols
    if args.crop_cols:
        default_model.param_dict['crop_cols'] = args.crop_cols
    if args.crop_rows:
        default_model.param_dict['crop_rows'] = args.crop_rows
    if args.pool_size_dim1 and args.pool_size_dim2:
        default_model.param_dict['pool_size'] = (args.pool_size_dim1, args.pool_size_dim2)
    if args.kernel_size_dim1 and args.kernel_size_dim2:
        default_model.param_dict['kernel_size'] = (args.kernel_size_dim1, args.kernel_size_dim2)
    if args.weights:
        default_model.param_dict['weights'] = args.weights

    default_model.train()
    m = model_one(default_model)
    m.autoencoder.fit(default_model.X_train,
                      default_model.X_crop_train,
                      batch_size=default_model.param_dict['batch_size'],
                      nb_epoch=default_model.param_dict['nb_epoch'],
                      verbose=1,
                      callbacks=[TensorBoard(log_dir='/tmp/autoencoder')],
                      validation_data=(default_model.X_test, default_model.X_crop_test))

    m.autoencoder.save_weights(default_model.param_dict['weights'])
    print("Saved model to disk")

def show_result(pred_result):
    result = pred_result.copy()
    result *= 255
    result = np.reshape(result,(result.shape[1],result.shape[1]))
    result = np.array(result,dtype='uint8')
    Image.fromarray(result).show()

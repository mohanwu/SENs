import crop_encoder_data
from keras import backend as K

class model_network_param():

  def __init__(self, **kwargs):
    self.param_dict = kwargs

  def train(self):
    self.X_train, self.X_crop_train, self.X_test, self.X_crop_test = crop_encoder_data.load_data(0.5)

    if K.image_dim_ordering() == 'th':
      self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.param_dict['img_rows'], self.param_dict['img_cols'])
      self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.param_dict['img_rows'], self.param_dict['img_cols'])
      self.X_crop_train = self.X_crop_train.reshape(self.X_train.shape[0], 1, self.param_dict['crop_rows'], self.param_dict['crop_cols'])
      self.X_crop_test = self.X_crop_test.reshape(self.X_test.shape[0], 1, self.param_dict['crop_rows'], self.param_dict['crop_cols'])
      #input_shape = (1, self.img_rows, img_cols)
    else:
      self.X_train = self.X_train.reshape(self.X_train.shape[0], self.param_dict['img_rows'], self.param_dict['img_cols'],1)
      self.X_test = self.X_test.reshape(self.X_test.shape[0], self.param_dict['img_rows'],self.param_dict['img_cols'], 1)
      self.X_crop_train = self.X_crop_train.reshape(self.X_train.shape[0], self.param_dict['crop_rows']*self.param_dict['crop_cols'])
      self.X_crop_test = X_crop_test.reshape(X_test.shape[0], self.param_dict['crop_rows']*self.param_dict['crop_cols'])
      #input_shape = (img_rows,img_cols,1)

    self.X_train = self.X_train.astype('float32')
    self.X_train /= 255
    self.X_test = self.X_test.astype('float32')
    self.X_test /= 255

    self.X_crop_train = X_crop_train.astype('float32')
    self.X_crop_train /= 255
    self.X_crop_test = X_crop_test.astype('float32')
    self.X_crop_test /= 255


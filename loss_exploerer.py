import numpy as np
import matplotlib.pyplot as plt
import crop_encoder_data
from skimage.measure import *
np.random.seed(1200)
im_size = (32,32)
scale = 800
n = 20
img_rows,img_cols = (128,128)
crop_rows,crop_cols = (32,32)
def mean_sq_error(v1,v2):
    err = np.linalg.norm(v1-v2)**2/v1.shape[0]
    return err

def mean_abs_error(v1,v2):
    err = np.linalg.norm(v1-v2)/v1.shape[0]
    return err

def norm_shape(v1):
    if(len(v1.shape) == 2):
        return np.reshape(v1,v1.shape[0]*v1.shape[1],1)
    else:
        return v1

def find_unit(target,dir,metric,scale_1=10,scale_2=2000):
    scale_1 = 10
    scale_2 = 2000

    return 1
X_train, X_crop_train , X_test , X_crop_test = crop_encoder_data.load_data(0.5)

X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols)
X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols)
X_crop_train = X_crop_train.reshape(X_train.shape[0], crop_rows*crop_cols)
X_crop_test = X_crop_test.reshape(X_test.shape[0], crop_rows*crop_cols)

random_vects = [np.random.rand(im_size[0]*im_size[1])*2 - np.ones(im_size[0]*im_size[1]) for x in range(n)]
random_vects = [scale*(vect/np.linalg.norm(vect)) for vect in random_vects]

test_img = X_crop_train[2]
print "image shape: "+str(test_img.shape)
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    err_img = test_img+random_vects[i]
    tr_img = np.array(test_img,dtype="uint8")
    tst_img = np.array(err_img,dtype="uint8")
    tr_img = np.reshape(tr_img,(32,32))
    tst_img = np.reshape(tst_img,(32,32))

    ax = plt.subplot(4, 5, i+1)

    plt.title(compare_ssim(tr_img,tst_img))
    plt.imshow((err_img).reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# print norm_shape(np.array([[1,2,3],[4,5,6],[7,8,9]])).shape
# print norm_shape(np.array([10,11,12])).shape
# print mean_sq_error(np.array([2,3,4]),np.array([1,2,3]))

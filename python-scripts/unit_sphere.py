import os
import numpy as np
import sklearn
import crop_encoder_data
import cv2
import matplotlib.pyplot as plt
parent_dir = os.path.dirname(os.getcwd())
path = parent_dir+'/train/cropped/ALB/img_00039.jpg'
sample_img = cv2.imread(path, 0)
radius = 2000

def L2(t, g):
  return np.linalg.norm(t-g)

def dist(t, g, loss_fn):
  return loss_fn(t, g)

def bin_srch(rad, low, high, loss_fn, target, rand_vec, eps):
  mid = (low + high) * 0.5
  low_vec = low * rand_vec
  mid_vec = mid * rand_vec
  high_vec = high * rand_vec
  low_d = dist(low_vec, target, loss_fn)
  mid_d = dist(mid_vec, target, loss_fn)
  high_d = dist(high_vec, target, loss_fn)
  print 'low_d:' + str(low_d)
  print 'mid_d:' + str(mid_d)
  print 'high_d:' + str(high_d)
  #print 'cur_dist:' + str(abs(mid_d - rad))
  if abs(mid_d - rad) < eps:
    return mid_vec
  else:
    if (float(low_d) > rad):
      print 'fuck'
      return None
    elif (float(low_d) < rad and float(mid_d) > rad):
      return bin_srch(rad, low, mid, loss_fn, target, rand_vec, eps)
    else:
      return bin_srch(rad, mid, high, loss_fn, target, rand_vec, eps)

def diff_search(rand_vec, loss_fn, target):
  err = list()
  scales = []
  for x in xrange(1, 20000):
    new_vec = rand_vec * x + target
    cur_d = dist(new_vec, target, loss_fn)
    scales.append(x)
    err.append(cur_d)
  centered_err = [abs(e-radius) for e in err]
  print str(min(centered_err)) + ' at ' + str(scales[centered_err.index(min(centered_err))])
  print min(centered_err)
  return (scales[centered_err.index(min(centered_err))]*rand_vec + target)


# returns a list of vectors that each
# represent an image that is 'dist' distance
# away from the original image
# loss_fn is the loss function we are using
# target is the original image (center of our circle)
# assume target is 32x32
# points is the number of points we want to use
def generate(target, points, loss_fn=L2, s1=0.000001, s2=10000):
  w = target.shape[0]
  h = target.shape[1]
  imgs = list()
  for x in xrange(points):
    rand_arr = np.random.rand(w, h)
    rand_arr = (1/np.linalg.norm(rand_arr)) * rand_arr
    s = diff_search(rand_vec=rand_arr, loss_fn=L2, target=target)
    #s = bin_srch(rad=1000.0, low=s1, high=s2, loss_fn=L2, target=target, rand_vec = rand_arr, eps=0.25)
    print "done: " + str(x)
    imgs.append(s)
  for (i,img) in enumerate(imgs):
    ax = plt.subplot((points/5) + 1, 5, i+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(img)
  ax = plt.subplot(3, 5, points+1)
  plt.title("radius = "+str(radius))
  plt.imshow(target)
  plt.show()

generate(target=sample_img, points=10)

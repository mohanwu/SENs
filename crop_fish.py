import scipy.misc
import os
import cv2
import numpy
import math
import json

img_loc = 'train/'
crop_loc = 'crop_coord/'
ALB = img_loc + 'ALB/' #replace alb with any other label to get the reuslts
#temp_dict = dict()


folders = ['ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT']
crop_files = ['alb_labels_ht.json', 'bet_labels.json', 'dol_labels.json', 'lag_labels.json', 'shark_labels.json', 'yft_labels.json']

def resize(d, s):
  for filename in os.listdir(img_loc + s):
    if filename.endswith('.jpg') and d.has_key(filename):
      sample_img = cv2.imread(ALB + filename)
      x_start = min(d[filename][0])
      x_end = max(d[filename][0])
      y_start = min(d[filename][1])
      y_end = max(d[filename][1])
      height, width, channels = sample_img.shape
      x_range = math.fabs(x_end - x_start) * 0.1;
      y_range = math.fabs(y_end - y_start) * 0.1;
      y_start = max(0, y_start - y_range)
      y_end = min(y_end + y_range, height)
      x_start = max(0, x_start - x_range)
      x_end = min(x_end + x_range, width)
      crop_img = sample_img[int(y_start):int(y_end), int(x_start):int(x_end)]
      cv2.imwrite(ALB + 'cropped/' + filename, crop_img)

def crop():
  temp_dict = dict()
  for x, y in zip(crop_files, folders):
    f = os.path.join(crop_loc, x)
    with open(f) as data_file:
      data = json.load(data_file)
      print type(data)
      for cur_entry in data:
        crop_info = cur_entry['annotations']
        x_ = list()
        y_ = list()
        f_name = cur_entry['filename']
        # should just be two sets of points (top left + bottom right)
        for points in crop_info:
          x_.append(points['x'])
          y_.append(points['y'])
        if len(y_) != 2 or len(x_) != 2:
          continue
        temp_dict[f_name] = (x_, y_)
        resize(temp_dict, y)

crop()




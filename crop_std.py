import scipy.misc
import os
import cv2
import numpy
import math
import json
from PIL import Image

img_loc = os.getcwd()+'/train/'
ALB = img_loc + 'ALB/' #replace alb with any other label to get the reuslts
ALB_dict = dict()
IM_NUM = 50

if os.path.isdir(os.getcwd()+"/train/cropped") == False:
    os.makedirs(os.getcwd()+"/train/cropped")
    fish_dirs = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for d in fish_dirs:
        os.makedirs(os.getcwd()+"/train/cropped/"+d)

def resize(d=ALB_dict):
  for filename in os.listdir(ALB):
    if filename.endswith('.jpg') and d.has_key(filename):
      sample_img = Image.open(ALB + filename)
      x_start = min(ALB_dict[filename][0])
      x_end = max(ALB_dict[filename][0])
      y_start = min(ALB_dict[filename][1])
      y_end = max(ALB_dict[filename][1])

      height, width = [sample_img.height,sample_img.width]
      x_range = math.fabs(x_end - x_start);
      y_range = math.fabs(y_end - y_start);
      big_range = max(x_range,y_range)
    #   if float(max(x_range,y_range)/min(x_range,y_range)) > 3 :
    #       continue
      translation = ((x_start + x_range)/2.0 - (x_start + big_range)/2.0, (y_start+y_range)/2.0 - (y_start + big_range)/2.0)
      y_start = max(0, y_start + translation[1])
      y_end = min(y_start + big_range, height)
      x_start = max(0, x_start + translation[0])
      x_end = min(x_start + big_range, width)
      crop_img = sample_img.crop((int(x_start),int(y_start),int(x_end),int(y_end)))
      #crop_img = sample_img[int(y_start - y_range):int(y_end + y_range), int(x_start - x_range):int(x_end + x_range)]
      crop_img = crop_img.resize((120,120), Image.ANTIALIAS)
      crop_img.save(img_loc + 'cropped/ALB/' + filename, "JPEG")
      print img_loc + 'cropped/ALB/' + filename

def crop():
  with open('alb_labels_ht.json') as data_file:
    data = json.load(data_file)
    print type(data)
    for cur_entry in data[:IM_NUM]:
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
      ALB_dict[f_name] = (x_, y_)

crop()
resize()

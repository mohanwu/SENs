import scipy.misc
import os
import cv2
import numpy
import math
import json

loc = 'train/ALB/cropped/'

hw = list()

for filename in os.listdir(loc):
    if filename.endswith('.jpg'):
      f = os.path.join(loc, filename)
      img = cv2.imread(f)
      if img is None:
        continue
      height, width = img.shape[:2]
      hw.append((height, width))

print sum([v[0] for v in hw])/float(len(hw))
print sum([v[1] for v in hw])/float(len(hw))
print max([v[0] for v in hw])
print min([v[0] for v in hw])
print max([v[1] for v in hw])
print min([v[1] for v in hw])


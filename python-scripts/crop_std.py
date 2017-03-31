import scipy.misc
import os
import cv2
import numpy
import math
import json
from PIL import Image
parent_dir = os.path.dirname(os.getcwd())
#Will crop IM_NUM # of images from targets folders
targets = ['ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT']
# targets = ['ALB']
IM_NUM = -1
image_size = (32,32)
# If IM_NUM is -1 then will do all images (except the last one)

if os.path.isdir(parent_dir+"/train/cropped") == False:
    os.makedirs(parent_dir+"/train/cropped")
    fish_dirs = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for d in fish_dirs:
        os.makedirs(parent_dir+"/train/cropped/"+d)
def deg_angle_between(x1,y1,x2,y2):
    from math import atan2, degrees, pi
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy,dx)
    rads %= 2*pi
    degs = degrees(rads)
    return(degs)

def get_rotated_cropped_fish(img,x1,y1,x2,y2):
    (h,w) = img.shape[:2]
    #calculate center and angle
    center = ( (x1+x2) / 2,(y1+y2) / 2)
    angle = np.floor(-deg_angle_between(x1,y1,x2,y2))
    #print('angle=' +str(angle) + ' ')
    #print('center=' +str(center))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))

    fish_length = np.sqrt((x1-x2)**2+(y1-y2)**2)
    cropped = rotated[(max((center[1]-fish_length/1.8),0)):(max((center[1]+fish_length/1.8),0)) ,
                      (max((center[0]- fish_length/1.8),0)):(max((center[0]+fish_length/1.8),0))]
    #imshow(img)
    #imshow(rotated)
    #imshow(cropped)
    resized = resize(cropped,(224,224))
    return(resized)

def crop_and_resize(target):
    print target
    img_loc = parent_dir+'/train/'
    ALB = img_loc + target+'/'
    ALB_dict = dict()
    with open(parent_dir+"/crop_coord/"+target.lower()+'_labels.json') as data_file:
        data = json.load(data_file)

    for cur_entry in data[:IM_NUM]:
        crop_info = cur_entry['annotations']
        x_ = list()
        y_ = list()
        f_name = cur_entry['filename']
        # should just be two sets of points (top left + bottom right)
        for points in crop_info:
            try:
                x_.append(points['x'])
                y_.append(points['y'])
            except:
                continue
            if len(y_) != 2 or len(x_) != 2:
                continue
            ALB_dict[f_name] = (x_, y_)

    for filename in os.listdir(ALB):
        if filename.endswith('.jpg') and ALB_dict.has_key(filename):
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
            y_start = max(0, y_start + translation[1]-big_range*0.1)
            y_end = min(y_start + big_range + big_range*0.1, height)
            x_start = max(0, x_start + translation[0]-big_range*0.1)
            x_end = min(x_start + big_range + big_range*0.1, width)
            crop_img = sample_img.crop((int(x_start),int(y_start),int(x_end),int(y_end)))
            #crop_img = sample_img[int(y_start - y_range):int(y_end + y_range), int(x_start - x_range):int(x_end + x_range)]
            crop_img = crop_img.resize(image_size, Image.ANTIALIAS)
            crop_img.save(img_loc + 'cropped/'+target+'/' + filename, "JPEG")
            print img_loc + 'cropped/'+target+'/' + filename

for target in targets:
    crop_and_resize(target)

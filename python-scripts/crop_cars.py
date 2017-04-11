"""
How to use:

Make sure to have the cars dataset downloaded and unzipped:
http://ai.stanford.edu/~jkrause/cars/car_dataset.html

(make sure to have training images / testing images and 'devkit' (contains the bounding boxes)

1. Navigate to where training images / testing images / devkit are contained
2. Place this script in the directory
3. Run this script "python crop_cars.py"

"""


import glob
import os
import scipy.io as sio
from PIL import Image
import sys
import shutil

if not (os.path.exists(os.path.join(os.getcwd()+"/devkit"))):
    print "You are not working in correct directory"
    sys.exit()


cwd = os.getcwd()
cars_train = os.path.join(os.getcwd() + "/cars_train")
cars_test = os.path.join(os.getcwd() + "/cars_test")
train = os.path.join(cwd + "/train")
test = os.path.join(cwd + "/test")

if not os.path.exists(train):
    print "Creating directories and moving around files.."
    train_folder = os.makedirs(train)
    test_folder = os.makedirs(test)
    shutil.move(cars_train,train)
    shutil.move(cars_test, test)

    cars_train = os.path.join(train + "/cars_train")
    os.makedirs(train + "/cars_train_crop")
    cars_train_crop = os.path.join(train + "/cars_train_crop")

    cars_test = os.path.join(test+ "/cars_test")
    os.makedirs(test + "/cars_test_crop")
    cars_test_crop = os.path.join(test + "/cars_test_crop")

else:
    print "{}  directory being created already exists.".format(train)
    sys.exit()


def build_coord_map():
    #Returns two dictionaries resembling dict[fname] = (x1,y1,x2,y2)
    train_name_to_coord = {}
    test_name_to_coord = {}

    devkit = os.path.join(os.getcwd() +"/devkit")
    cars_train_annos = os.path.join(devkit + "/cars_train_annos.mat")
    train_crop_mat = sio.loadmat(cars_train_annos)
    train_crop_coords = train_crop_mat["annotations"]

    for i in range(len(train_crop_coords[0])):
        x1 = train_crop_coords[0,i][0][0][0]
        y1 = train_crop_coords[0,i][1][0][0]
        x2 = train_crop_coords[0,i][2][0][0]
        y2 = train_crop_coords[0,i][3][0][0]
        class_num = train_crop_coords[0,i][4][0][0]
        fname = train_crop_coords[0,i][5][0]
        train_name_to_coord[fname] = (x1,y1,x2,y2)

    cars_test_annos = os.path.join(devkit + "/cars_test_annos.mat")
    test_crop_mat = sio.loadmat(cars_test_annos)
    test_crop_coords = test_crop_mat["annotations"]

    for i in range(len(test_crop_coords[0])):
        x1 = test_crop_coords[0,i][0][0][0]
        y1 = test_crop_coords[0,i][1][0][0]
        x2 = test_crop_coords[0,i][2][0][0]
        y2 = test_crop_coords[0,i][3][0][0]
        fname = test_crop_coords[0,i][4][0]
        test_name_to_coord[fname] = (x1,y1,x2,y2)

    return train_name_to_coord, test_name_to_coord

train_name_to_coords, test_name_to_coords = build_coord_map()

def get_files(src,dest,name_to_coords):
    print "Cropping images from: {}".format(src)
    print "Moving cropped images to: {}".format(dest)

    imgs = os.listdir(src)

    if (not os.path.exists(src)) or (not os.path.exists(dest)):
        print "One of src: {}; or dest: {} doesn't exist".format(src,dest)
        sys.exit()

    count = 0
    num_imgs = len(imgs)
    print "Starting to crop {} images..".format(num_imgs)
    for img in imgs:
        coords = name_to_coords[img]
        name = img[::-1][4:][::-1]
        img_path = os.path.join(src + "/{}".format(img))
        x1 = coords[0]
        y1 = coords[1]
        x2 = coords[2]
        y2 = coords[3]
        image = Image.open(img_path)
        crop = image.crop((x1,y1,x2,y2))
        crop_path = os.path.join(dest +"/{}".format(name) + "_crop.jpg")
        crop.save(crop_path, "JPEG")
        count += 1

    print "Cropped {}/{} images".format(count,num_imgs)


get_files(str(cars_train),str(cars_train_crop),train_name_to_coords)

print "--------------------------------"
print "--------------------------------"

get_files(str(cars_test),str(cars_test_crop),test_name_to_coords)



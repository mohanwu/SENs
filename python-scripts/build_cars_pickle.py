import glob
import os
import scipy.io as sio
from PIL import Image
import sys
import shutil
import urllib2
import tarfile
import cv2
import math
import glob
import numpy as np
import pickle



if os.path.exists(str(os.getcwd()+'/cars_img_data.pkl')):
    print "cars pickle file already exists, exiting script"
    sys.exit()


train_url = "http://imagenet.stanford.edu/internal/car196/cars_train.tgz"
test_url = "http://imagenet.stanford.edu/internal/car196/cars_test.tgz"
devkit_url = "http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"

def extract_tar(file_name):
    tar = tarfile.open(file_name, "r:gz")
    tar.extractall()
    tar.close()
    print "Finished extracting {}".format(file_name)

def download_from_url(url):
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "---------------------------------------"
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"[%3.2f%%]" % (file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()

    extract_tar(file_name)

#download_from_url(devkit_url)
#download_from_url(train_url)
#download_from_url(test_url)

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
    """
    src: path of source images
    dest: path of where cropped images should end up
    name_to_coords: dictionary that maps image name to bounding box coords
    """
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
get_files(str(cars_test),str(cars_test_crop),test_name_to_coords)

train = os.path.join(cwd + "/train")
test = os.path.join(cwd + "/test")

if os.path.exists(train):
    cars_train_gs = os.path.join(train + "/cars_train_gs")
    os.makedirs(train + "/cars_train_gs")
    cars_train_crop_gs = os.path.join(train+ "/cars_train_crop_gs")
    os.makedirs(train + "/cars_train_crop_gs")

    cars_test_gs = os.path.join(test + "/cars_test_gs")
    os.makedirs(test + "/cars_test_gs")
    cars_test_crop_gs = os.path.join(test+ "/cars_test_crop_gs")
    os.makedirs(test + "/cars_test_crop_gs")

train = os.path.join(cwd + "/train")
test = os.path.join(cwd + "/test")

cars_train = os.path.join(train + "/cars_train")
cars_train_crop = os.path.join(train + "/cars_train_crop")
cars_test = os.path.join(test + "/cars_test")
cars_test_crop = os.path.join(test + "/cars_test_crop")
cars_train_gs = os.path.join(train + "/cars_train_gs")
cars_train_crop_gs = os.path.join(train+ "/cars_train_crop_gs")
cars_test_gs = os.path.join(test + "/cars_test_gs")
cars_test_crop_gs = os.path.join(test+ "/cars_test_crop_gs")

cars_train = str(cars_train)
cars_train_gs = str(cars_train_gs)
cars_train_crop = str(cars_train_crop)
cars_train_crop_gs = str(cars_train_crop_gs)

cars_test = str(cars_test)
cars_test_gs = str(cars_test_gs)
cars_test_crop = str(cars_test_crop)
cars_test_crop_gs = str(cars_test_crop_gs)

def greyscale_imgs(src, dest):
    print "--------------------------------------------------"
    print "Greyscaling images in: {}".format(src)
    print "Moving greyscaled images to: {}".format(dest)

    imgs = glob.glob(src+'/*.jpg')

    if (not os.path.exists(src)) or (not os.path.exists(dest)):
        print "One of src: {}; or dest: {} doesn't exist".format(src,dest)
        sys.exit()

    count = 0
    num_imgs = len(imgs)
    print "Starting to greyscale {} images..".format(num_imgs)
    for img in imgs:
        name = img.split('.')[0]
        name = name.split('/')[-1]
        image = Image.open(img)
        image = image.convert('L')
        blk_white_path = os.path.join(dest +"/{}".format(name) + "_greyscale.jpg")
        image.save(blk_white_path, "JPEG")
        count += 1

    print "Greyscaled {}/{} images".format(count,num_imgs)

greyscale_imgs(cars_train, cars_train_gs)
greyscale_imgs(cars_train_crop, cars_train_crop_gs)
greyscale_imgs(cars_test, cars_test_gs)
greyscale_imgs(cars_test_crop, cars_test_crop_gs)

train = os.path.join(cwd + "/train")
test = os.path.join(cwd + "/test")
cars_train_std = os.path.join(train + "/cars_train_std")
cars_train_crop_std = os.path.join(train + "/cars_train_crop_std")
cars_test_std = os.path.join(test + "/cars_test_std")
cars_test_crop_std = os.path.join(test + "/cars_test_crop_std")

if not os.path.exists(cars_train_std):
    os.makedirs(train + "/cars_train_std")
    os.makedirs(train + "/cars_train_crop_std")
    os.makedirs(test + "/cars_test_std")
    os.makedirs(test + "/cars_test_crop_std")

cars_train_std = str(cars_train_std)
cars_train_crop_std = str(cars_train_crop_std)
cars_test_std = str(cars_test_std)
cars_test_crop_std = str(cars_test_crop_std)

def resize_imgs(src, dest, sz):
    print "--------------------------------------------------"
    print "Resizing images in: {}".format(src)
    print "Moving resized images to: {}".format(dest)

    imgs = glob.glob(src+'/*.jpg')

    if (not os.path.exists(src)) or (not os.path.exists(dest)):
        print "One of src: {}; or dest: {} doesn't exist".format(src,dest)
        sys.exit()

    count = 0
    num_imgs = len(imgs)
    print "Starting to resize {} images..".format(num_imgs)
    for img in imgs:
        name = img.split('.')[0]
        name = name.split('/')[-1]
        image = Image.open(img)
        image = image.resize((sz,sz), Image.ANTIALIAS)
        path = os.path.join(dest +"/{}".format(name) + "_std.jpg")
        image.save(path, "JPEG")
        count += 1

    print "Resized {}/{} images".format(count,num_imgs)

resize_imgs(cars_train_gs, cars_train_std, 128)
resize_imgs(cars_train_crop_gs, cars_train_crop_std, 32)
resize_imgs(cars_test_gs, cars_test_std, 128)
resize_imgs(cars_test_crop_gs, cars_test_crop_std, 32)

def build_pickle(train_std_path, train_crop_std_path, test_std_path, test_crop_std_path):
    train_std_imgs = glob.glob(train_std_path+'/*.jpg')
    train_crop_std_imgs = glob.glob(train_crop_std_path+'/*.jpg')
    test_std_imgs = glob.glob(test_std_path+'/*.jpg')
    test_crop_std_imgs = glob.glob(test_crop_std_path+'/*.jpg')
    X_train = map(lambda f_name: np.asarray(Image.open(f_name), dtype='uint8'), train_std_imgs)
    y_train = map(lambda f_name: np.asarray(Image.open(f_name), dtype='uint8'), train_crop_std_imgs)
    X_test = map(lambda f_name: np.asarray(Image.open(f_name), dtype='uint8'), test_std_imgs)
    y_test = map(lambda f_name: np.asarray(Image.open(f_name), dtype='uint8'), test_crop_std_imgs)
    img_data = {}
    img_data['X_train'] = X_train
    img_data['y_train'] = y_train
    img_data['X_test'] = X_test
    img_data['y_test'] = y_test
    with open('cars_img_data.pkl', 'wb') as my_pickle:
        pickle.dump(img_data,my_pickle)
    print "Finished building pickle"

build_pickle(cars_train_std, cars_train_crop_std, cars_test_std, cars_test_crop_std)
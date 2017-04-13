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

def load_pickle(pickle_path):
    if not os.path.exists(pickle_path):
        print "either pickle file DOESN'T EXISTS, or not in the same directory as pickle"
        print "run build_cars_pickle.py first. exiting script"
        sys.exit()
    with open(pickle_path,"rb") as img_f:
        data = pickle.load(img_f)

    return (data['X_train'],data['y_train'],data['X_test'],data['y_test'])

X_train, y_train, X_test, y_test = load_pickle(str(os.getcwd()+'/cars_img_data.pkl'))


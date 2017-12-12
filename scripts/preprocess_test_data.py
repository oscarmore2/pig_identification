import cv2
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import time
from datetime import datetime
import os
import random as rand
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plot
#from utils.data_utils import *
from utils.vis_utils import *
from utils.layer_utils import *
from utils.print_utils import *
from utils.resnet import *
import pickle
import problem_unittests as tests
import helper
from glob import glob

def file_name(file_dir):
    L=[]
    num=[]
    for root, dirs, files in os.walk(file_dir):
        #print(files)
        for file in files:
            if os.path.splitext(file)[1] == '.JPG':
                p = os.path.join(root, file)
                n = os.path.splitext(file)[0]
                yield p, n
    #return L, num
    #
def loadImgsAndSave():
    _x_ = []
    paths = []
    for path, num in file_name('../Pig_Identification_Qualification_Train/test_set'):
        _x = cv2.imread(path)[:,:,::-1]
        pImg = Image.fromarray(_x, mode='RGB')
        if _x.shape[0] > _x.shape[1]:
            pImg.transpose(Image.ROTATE_90)
        pImg = pImg.resize((64,120))
        _x_.append(np.array(pImg))
        paths.append(num)
        #_y_.append(Label_OneHot(1, rand.randint(1, 30)))
    result = {'data':np.array(_x_), 'path':paths}
    f = open('test_set_120.p', 'wb')
    pickle.dump((result), f)
    f.close()

loadImgsAndSave()
print('finish preprocessing test data')
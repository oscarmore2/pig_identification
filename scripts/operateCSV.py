import csv
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

nump = []
with open('./output1.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        nump.append(row)

nump = np.cast['float'](np.array(nump))
kkk = []
for jjj in range(int(nump.shape[0] / 30)):
    lll = nump[jjj * 30:jjj * 30 + 30]
    #print(lll)
    lll.sort(axis=0)
    for eles in range(lll.shape[0]):
        kkk.append(lll[eles])
kkk = np.array(kkk)
print(kkk[:35])

print(kkk.shape)
with open('./output3_new.csv', 'w') as w:
    writer = csv.writer(w)
    for jjs in range(kkk.shape[0]):
        writer.writerow([str(int(kkk[jjs][0])), str(int(kkk[jjs][1])), '{:.9f}'.format(kkk[jjs][2])])
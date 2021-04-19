import Make_wavedata, clusteringPoint, savePict
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from pythonFile import click_pct, k_means, timestump, getVideoData
import math
from tkinter import filedialog
import scipy.stats
import os
import time
import pickle

# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = '/media/koshiba/Data/video'
path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
time_data = timestump.get_time()
start = time.time()

dirName = getVideoData.getDirName(path)
videoName = getVideoData.getVideoName(path)

f = open('/media/koshiba/Data/opticalflow/point_data/' + dirName + '/' + videoName + '/category.txt', 'rb')
noise = pickle.load(f)

#noise = clusteringPoint.todo(path)
classList = Make_wavedata.todo(path, time_data)
print(noise)

predList = [[],[],[]]
accuracy = ['-1', '-1', '-1']
precision = ['-1', '-1', '-1']
recall = ['-1', '-1', '-1']
specificity = ['-1', '-1', '-1']
tmp = 0

for index1, pred in enumerate(classList):
    for index2, answer in enumerate(noise):
        #print(index1, index2)
        if (pred[index2]==0 or pred[index2]==-1):
            if answer==0:
                predList[index1].append(0)
            else:
                predList[index1].append(3)
        else:
            if answer==1:
                predList[index1].append(1)
            else:
                predList[index1].append(2)
    
    predAll = len(predList[index1])
    tp = predList[index1].count(1)
    tn = predList[index1].count(0)
    fp = predList[index1].count(2)
    fn = predList[index1].count(3)

    print(predAll, tp, tn)
    accuracy[index1] = (tp + tn)/predAll
    if tp+fp != 0:
        precision[index1] = tp/(tp+fp)
    if tp+fn != 0:
        recall[index1] = tp/(tp+fn)
    if tn+fp != 0:
        specificity[index1] = tn/(fp+tn)

print(classList)

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

print('acc  : {:.3f} | {:.3f} | {:.3f}'.format(accuracy[0],accuracy[1],accuracy[2]))
print('pre  : {:.3f} | {:.3f} | {:.3f}'.format(precision[0],precision[1],precision[2]))
print('rec  : {:.3f} | {:.3f} | {:.3f}'.format(recall[0],recall[1],recall[2]))
print('spe  : {:.3f} | {:.3f} | {:.3f}'.format(specificity[0],specificity[1],specificity[2]))

savePict.todo(path, classList, noise)

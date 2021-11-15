import Make_wavedata, clusteringPoint, savePict
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from pythonFile import click_pct, k_means, timestump
import math
from tkinter import filedialog
import scipy.stats
import os
import time

# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = 'C:\\pg'
path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
time_data = timestump.get_time()
start = time.time()

noise = clusteringPoint.todo(path)
classList = Make_wavedata.todo(path, time_data)
savePict.todo(path, classList)
print(noise)

predList = [[],[],[]]
accuracy = ['-1', '-1', '-1']
precision = ['-1', '-1', '-1']
recall = ['-1', '-1', '-1']
specificity = ['-1', '-1', '-1']
tmp = 0
for index1, pred in enumerate(classList):
    for index2, answer in enumerate(noise):
        if answer==pred[index2]:
            predList[index1].append(answer)
        else:
            predList[index1].append(answer+2)
    
    predAll = len(predList[index1])
    tp = predList[index1].count(1)
    tn = predList[index1].count(0)
    fp = predList[index1].count(2)
    fn = predList[index1].count(3)

    print(predAll, tp, tn)
    accuracy[index1] = str((tp + tn)/predAll)
    if tp+fp != 0:
        precision[index1] = str(tp/(tp+fp))
    if tp+fn != 0:
        recall[index1] = str(tp/(tp+fn))
    if tn+fp != 0:
        specificity[index1] = str(tn/(fp+tn))

print(classList)

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

print('accuracy:' + accuracy[0] + ' ' + accuracy[1] + ' '  + accuracy[2])
print('precision' + precision[0] + ' '  + precision[1] + ' '  + precision[2])
print('recall' + recall[0] + ' '  + recall[1] + ' '  + recall[2])
print('specificity' + specificity[0] + ' '  + specificity[1] + ' '  + specificity[2])

savePict.todo(path, classList, noise)


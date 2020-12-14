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
typ = [('','*')] 
dir = 'C:\\pg'
path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
cap = cv2.VideoCapture(path)
bgs = cv2.bgsegm.createBackgroundSubtractorGSOC()
videoName = path[path.rfind('/')+1:]
i = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    i += 1
    mask = bgs.apply(frame)
    bg = bgs.getBackgroundImage()
    cv2.imshow('mask', mask)
    cv2.imshow('bg', bg)
    cv2.imwrite("D:/opticalflow/evaluation/data/" + str(i) + ".jpg",mask)
    if cv2.waitKey(1) != -1:
        break

cap.release()
cv2.destroyAllWindows()
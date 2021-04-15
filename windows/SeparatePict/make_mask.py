import numpy as np
import cv2
import matplotlib.pyplot as plt
#import torch
from tkinter import filedialog
import os
#import torchvision
#from torchvision import transforms
from pythonFile import getVideoData


if __name__ == '__main__':
    
    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir) 
    img = cv2.imread(image_path)

    color_palette = [0, 0, 0, 255, 255, 255]

    #saveImg =  np.zeros_like(img)
    '''
    for ni, i in enumerate(img):
        for nj, j in enumerate(i):
            if not sum(j)==0:
                print('hello')
                saveImg[ni][nj] = [255, 255, 255]
    '''
    img = np.where(img.sum(axis=2) > 0, 255, 0)
    
    #with Image.fromarray()
    cv2.imwrite('D:/opticalflow/mask/test1.jpg', img)
    #print(saveImg[200])

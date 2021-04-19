import numpy as np
import cv2
import matplotlib.pyplot as plt
#import torch
from tkinter import filedialog
import glob
import os
#import torchvision
#from torchvision import transforms
from pythonFile import getVideoData


if __name__ == '__main__':
    
    # ファイルダイアログからファイル選択
    typ = [('','*')] 
    dir = 'C:\\pg'
    image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
    image_file = glob.glob(image_path[:image_path.rfind('/')] + '/*')
    image_dir = image_path.split('/')[-2] + '/'
    if not os.path.exists('D:/opticalflow/mask/' + image_dir):
        os.makedirs('D:/opticalflow/mask/' + image_dir)
    for image in image_file:
        image = image.replace(os.sep, '/')
        print(image)
        image_name = image.split('/')[-1][:image.split('/')[-1].rfind('.')]
        img = cv2.imread(image)

        #color_palette = [0, 0, 0, 255, 255, 255]

        img = np.where(img.sum(axis=2) > 0, 255, 0)
        
        #with Image.fromarray()
        cv2.imwrite('D:/opticalflow/mask/' + image_dir + image_name + '.png', img)
        #print(saveImg[200])
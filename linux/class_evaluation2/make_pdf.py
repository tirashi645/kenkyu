import img2pdf
import os
from PIL import Image
from pythonFile import getVideoData
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tkinter import filedialog
import glob
import shutil

typ = [('','*')] 
if os.path.isdir('/media/koshiba/Data/opticalflow/point_data'):
    dir = '/media/koshiba/Data/opticalflow/point_data'
else:
    dir = 'C:\\pg'
path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
dirName = getVideoData.getDirName(path)
videoName = getVideoData.getVideoName(path)
dirPath = path[:path.rfind(dirName)+len(dirName)]

pdf_FileName = dirPath + '/output.pdf' # 出力するPDFの名前

for i in range(2):
    if i == 0:
        fileList = glob.glob(dirPath + '/**/winsize/average.jpg')
    else:
        pdf_FileName = dirPath + '/outputAll.pdf' # 出力するPDFの名前
        fileList = glob.glob(dirPath + '/**/winsize/average*.jpg')

    with open(pdf_FileName,"wb") as f:
        f.write(img2pdf.convert([Image.open(data).filename for data in fileList]))
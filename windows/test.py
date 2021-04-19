from PIL import Image
from tkinter import filedialog
import glob
import cv2
import os


# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
image_file = glob.glob(image_path[:image_path.rfind('/')] + '/*')

for image in image_file:
    image = image.replace(os.sep, '/')
    image_name = image.split('/')[-1][:image.split('/')[-1].rfind('.')]
    print(image_name)
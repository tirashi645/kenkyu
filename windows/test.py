from PIL import Image
from tkinter import filedialog
import glob
import cv2
import os
import pickle

'''
# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

with open(image_path, 'rb') as f:
    data = pickle.load(f)

print(data)
'''

n = 1000
n /= 10
print(n)
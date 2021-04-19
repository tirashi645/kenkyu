import os
import shutil
import glob
from tkinter import filedialog

typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
data = glob.glob('D:/opticalflow/mask/**/*.png')
for i in data:
    i = i.replace(os.sep, '/')
    dir_name = i.split('/')[-2]
    img_name = i.split('/')[-1]
    if not os.path.exists('D:/opticalflow/pict_all/mask/' + dir_name + '_' + img_name[:img_name.rfind('.')] + '.jpg'):
        shutil.copy(i, 'D:/opticalflow/pict_all/mask/' + dir_name + '_' + img_name[:img_name.rfind('.')] + '.jpg')
    if not os.path.exists('D:/opticalflow/pict_all/pict/' + dir_name + '_' + img_name[:img_name.rfind('.')] + '.jpg'):
        shutil.copy('D:/opticalflow/pict/' + dir_name + '/' + img_name[:img_name.rfind('.')] + '.jpg', 'D:/opticalflow/pict_all/pict/' + dir_name + '_' + img_name[:img_name.rfind('.')] + '.jpg')
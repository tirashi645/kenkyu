from PIL import Image
from tkinter import filedialog
import glob
import cv2
import os

# ファイルダイアログからファイル選択
'''
typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
'''
image_list = glob.glob('D:/opticalflow/pingFile/myVideo/**/*')
file1 = glob.glob('E:/data/background/*')

for image_path in image_list:
    for i, data in enumerate(file1):
        frame = cv2.imread(data)
        png_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # アルファチャンネル込みで読み込む
        image_path = image_path.replace(os.sep, '/')
        image_dir = image_path.split('/')[-2]
        image_name = image_path.split('/')[-1][:image_path.split('/')[-1].rfind('.')]

        # 貼り付け先座標の設定。とりあえず左上に
        x1, y1, x2, y2 = 0, 0, png_image.shape[1], png_image.shape[0]

        # 合成!
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - png_image[:, :, 3:] / 255) + \
                            png_image[:, :, :3] * (png_image[:, :, 3:] / 255)
        if not os.path.exists('E:/data/synthetic/' + image_dir + '_' + image_name + '_' + str(i) + '.jpg'):
            cv2.imwrite('E:/data/synthetic/' + image_dir + '_' + image_name + '_' + str(i) + '.jpg', frame)
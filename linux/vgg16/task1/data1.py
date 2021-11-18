import os
import glob
import shutil
import cv2

filePath = glob.glob('/media/koshiba/Data/sport/**/**/*.jpg')
for i, file in enumerate(filePath):
    print(file)
    img = cv2.imread(file)
    height,width = img.shape[:2]

    target_size = (height,width) #src size < dst sizeの前提

    top = int((target_size[1] - height)/2)
    bottom = target_size[1] - height - top
    left = int((target_size[0] - width)/2)
    right = target_size[0] - width - left

    color = (255,255,255)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,cv2.BORDER_CONSTANT,value=color)
    img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(file, img)
import os
import glob
import shutil
import cv2

filePath = glob.glob('/media/koshiba/Data/sport/**/**/*.jpg')
for i, file in enumerate(filePath):
    print(file)
    im = cv2.imread(file)
    im = cv2.resize(im, (150, 150), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('/media/koshiba/Data/sport/**/**/*.jpg', im)
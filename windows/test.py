from PIL import Image
from tkinter import filedialog
import glob
import os
import pickle
import numpy as np
import cv2

# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

cap = cv2.VideoCapture(image_path)
wait_secs = int(1000 / cap.get(cv2.CAP_PROP_FPS))

model = cv2.bgsegm.createBackgroundSubtractorMOG()

hog = cv2.HOGDescriptor()
hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)

# SVMによる人検出
hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

'''
# リサイズした方が精度がよかった
finalHeight = 800.0
scale = finalHeight / image.shape[0]
image = cv2.resize(image, None, fx=scale, fy=scale)
'''
while True:
    ret, image = cap.read()
    if not ret:
        break

    #mask = model.apply(frame)
    # リサイズした方が精度がよかった
    finalHeight = 800.0
    scale = finalHeight / image.shape[0]
    image = cv2.resize(image, None, fx=scale, fy=scale)
    # 人を検出した座標
    human, r = hog.detectMultiScale(image, hitThreshold = 0.6, winStride = (8,8), padding = (32, 32), scale = 1.05, finalThreshold=2)
    # 全員のバウンディングボックスを作成
    for (x, y, w, h) in human:
        cv2.rectangle(image, (x, y),(x+w, y+h),(0,255,0), 2)

    cv2.imshow("Mask", image)
    cv2.waitKey(wait_secs)

cap.release()
cv2.destroyAllWindows()
import cv2
import sys
import numpy as np
from tkinter import filedialog
 
def nothing(x):
     pass

# ファイルダイアログからファイル選択
typ = [('','*')] 
dir = 'C:\\pg'
image_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
file_path = image_path
#file_path = 'back1_1.mp4'
delay = 1
window_name = 'frame'
 
cap = cv2.VideoCapture(file_path)
 
# flag = 4
# detector = None
# if flag==1:
#     # AgastFeatureDetector
#     detector = cv2.AgastFeatureDetector_create()
# elif flag==2:
#     # FAST
#     detector = cv2.FastFeatureDetector_create()
# elif flag==3:
#     # MSER
#     detector = cv2.MSER_create()
# elif flag==4:
#     # AKAZE
#     detector = cv2.AKAZE_create()
# elif flag==5:
#     # BRISK
#     detector = cv2.BRISK_create()
# elif flag==6:
#     # KAZE
#     detector = cv2.KAZE_create()
# elif flag==7:
#     # ORB (Oriented FAST and Rotated BRIEF)
#     detector = cv2.ORB_create()
# elif flag==8:
#     # SimpleBlobDetector
#     detector = cv2.SimpleBlobDetector_create()
# else:
#     # SIFT
#     detector = cv2.xfeatures2d.SIFT_create()
 
# fast = cv2.FastFeatureDetector_create()
 
if not cap.isOpened():
    sys.exit()
 
cv2.namedWindow(window_name)
cv2.namedWindow('slider')
cv2.createTrackbar('NumOfPoints','slider', 255, 255, nothing)
cv2.createTrackbar('qualityLevel','slider', 1, 50, nothing)
cv2.createTrackbar('minDistance','slider',  20, 100, nothing)
cv2.createTrackbar('clockSize','slider',  3, 10, nothing)
 
num = 100

while True:
    num = cv2.getTrackbarPos('NumOfPoints', 'slider')
    num2 = cv2.getTrackbarPos('qualityLevel', 'slider')
    num3 = cv2.getTrackbarPos('minDistance', 'slider')
    num4 = cv2.getTrackbarPos('blockSize', 'slider')
    ret, frame = cap.read()
    if ret:
        gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # コーナー検出
        #corners = cv2.goodFeaturesToTrack(gray1, num, 0.01, 20)
        corners = cv2.goodFeaturesToTrack(gray1, num, num2 / 100, num3, num4)
        print(num, num2 / 100, num3)

        # print('----')
        # print(type(corners))
        # print('#####')
        if not (corners is None): 
            corners = np.int0(corners)
 
            # 特徴点を元画像に反映
            for i in corners:
                x, y = i.ravel()
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
 
        # kp1 = detector.detect(gray1)
        # img1_fast = cv2.drawKeypoints(gray1, kp1, None, flags=4)
 
        cv2.imshow(window_name, frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
 
cv2.destroyWindow(window_name)
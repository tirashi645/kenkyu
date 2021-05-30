from tkinter import filedialog
import cv2
import glob

file1 = glob.glob('D:/opticalflow/pingFile/myVideo/**/*')

for path in file1:
    print(path)
    data = cv2.imread(path, -1)
    data = cv2.resize(data, (720, 1280))
    cv2.imwrite(path, data)
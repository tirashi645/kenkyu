import numpy as np
import cv2
from tkinter import filedialog

typ = [('','*')] 
dir = 'C:\\pg'
path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

videoName = path[path.rfind('/')+1:]
cap = cv2.VideoCapture(path)
print(videoName)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))+10
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))+10
fps = cap.get(cv2.CAP_PROP_FPS)
rot = 0
print(width, height)
if width>height:
    rot = 1
    tmp = width
    width = height
    height = tmp

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
writer = cv2.VideoWriter('D:/opticalflow/tmp_video/'+ videoName[:-4] + '.avi', fourcc, fps, (width, height))

ret, first_frame = cap.read()
if rot==1:
    first_frame = np.rot90(first_frame, -1)



first_frame_pad = np.pad(first_frame, [(5,5),(5,5),(0,0)], 'constant')

writer.write(first_frame_pad)

while(True):
    ret, frame = cap.read()
    if ret!=True:
        break
    if rot==1:
        frame = np.rot90(frame, -1)
    frame_pad = np.pad(frame, [(5,5),(5,5),(0,0)], 'constant')
    writer.write(frame_pad)
    print(frame_pad.shape, first_frame_pad.shape)

cap.release()
writer.release()
cv2.destroyAllWindows()

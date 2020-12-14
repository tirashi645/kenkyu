import numpy as np
import cv2
import glob
import os

print("inputData>>")
inPath = input()
print("outputData>>")
outPath = input()
print(outPath)

videoDir = inPath
videolist = glob.glob(videoDir + "/*")

for i in videolist:
    print(type(i))
    cap = cv2.VideoCapture(i)
    videoName = i[i.rfind("/")+1:]
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
    writer = cv2.VideoWriter(outPath + '/' + videoName[:-4] + '.avi', fourcc, fps, (width, height))
    print(outPath + '/' + videoName[:-4] + '.avi')

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
        #print(frame_pad.shape, first_frame_pad.shape)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print('finish')
